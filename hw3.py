import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'

N = 2000


def shiftMatr(vec):
    mtr = np.array([[1, 0, vec[0]], [0, 1, vec[1]], [0, 0, 1]])
    return mtr


def rotMatr(ang):
    mtr = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    return mtr


def sizeMatr(koef):
    mtr = np.array([[koef, 0, 0], [0, koef, 0], [0, 0, 1]])
    return mtr


def to_proj_coords(arr):
    r, lenght = arr.shape
    arr = np.concatenate([arr, np.ones((1, lenght))], axis=0)
    return arr


def to_cart_coords(x):
    x = x[:-1] / x[-1]
    return x


def unpack(file):
    vertex = []
    face = []
    with open(file, "r") as f:
        for s in f:
            spl_str = s.split()
            if not spl_str:  # pass empty string
                continue

            if spl_str[0] == 'v':
                for i in range(1, len(spl_str)):  # type conversion (str->float)
                    spl_str[i] = float(spl_str[i])
                vertex.append(spl_str[1:])
            elif spl_str[0] == 'f':
                for i in range(1, len(spl_str)):  # type conversion  (str->int)
                    spl_str[i] = int(spl_str[i])
                face.append(spl_str[1:])
    return vertex, face


def change_xy(vertex):
    size = 1024
    vertexxy = np.array(vertex)
    vertexxy = vertexxy[:, 0:2]
    max_x = max(vertexxy[i][0] for i in range(len(vertexxy)))
    max_y = max(vertexxy[i][1] for i in range(len(vertexxy)))
    mmax = max(max_x, max_y)

    vertexxy[:, 0] /= mmax
    vertexxy[:, 1] /= mmax
    vertexxy[:, 0:2] *= (size / 2 - 1)

    # сдвиг в центр (центр чайника - донышко)
    vertexxy[:, 0:2] += N / 2

    # переворот чайника
    r = 0
    for ver in vertexxy:
        r = ver[1] - N / 2
        ver[1] -= r * 2

    # сдвиг в центр (=центр чайника)
    max_y = max(vertexxy[i][1] for i in range(len(vertexxy)))
    min_y = min(vertexxy[i][1] for i in range(len(vertexxy)))
    max_x = max(vertexxy[i][0] for i in range(len(vertexxy)))
    min_x = min(vertexxy[i][0] for i in range(len(vertexxy)))
    med_y = (np.abs(min_y) + np.abs(max_y)) / 2
    med_x = (np.abs(min_x) + np.abs(max_x)) / 2
    vertexxy[:, 1] += (N / 2 - med_y)
    vertexxy[:, 0] += (N / 2 - med_x)

    return vertexxy


def draw(vertexxy, face, color):
    img = np.zeros((N, N, 3), dtype=np.uint8) + 255

    for trian in face:
        # прорисовываем все 3 ребра
        line1 = alg_Br(vertexxy[trian[0] - 1][0], vertexxy[trian[0] - 1][1],
                       vertexxy[trian[1] - 1][0], vertexxy[trian[1] - 1][1])
        line2 = alg_Br(vertexxy[trian[1] - 1][0], vertexxy[trian[1] - 1][1],
                       vertexxy[trian[2] - 1][0], vertexxy[trian[2] - 1][1])
        line3 = alg_Br(vertexxy[trian[0] - 1][0], vertexxy[trian[0] - 1][1],
                       vertexxy[trian[2] - 1][0], vertexxy[trian[2] - 1][1])

        # !img[номер строки (по y), номер столбца (по x)]
        for pixel in line1:
            img[pixel[1], pixel[0]] = color
        for pixel in line2:
            img[pixel[1], pixel[0]] = color
        for pixel in line3:
            img[pixel[1], pixel[0]] = color

    return img


def make_frames(vertexxy, face):
    center = np.array([N/2,N/2])
    frames = []
    fig = plt.figure()

    k = 50
    for round in range(2):
        for i in range(k):
            # change color and size
            delta_color = (255/k)*i

            if round == 0:  # uneven
                color = np.array([255-int(delta_color),0,0+int(delta_color)],dtype=np.uint8) # to blue
                delta_size = 1 + i / k # х2
                S = sizeMatr(delta_size)
            else:  # even
                color = np.array([0+int(delta_color), 0, 255-int(delta_color)], dtype=np.uint8) # to red
                delta_size = 2 - i / k
                S = sizeMatr(delta_size)

            # change koordinats
            vert_proj = to_proj_coords(vertexxy.T)

            ang = i * 2 * np.pi / k
            T = shiftMatr(center * (-1))
            R = rotMatr(ang)
            vert_proj_new = np.linalg.inv(T) @ S @ R @ T @ vert_proj

            vertex_new = to_cart_coords(vert_proj_new)
            vertex_new_T = vertex_new.T

            # drawing
            img = draw(vertex_new_T, face, color)
            frames.append([plt.imshow(img)])

    return frames,fig


# формировка анимации из набора кадров
def create_anim(frames,fig):
    # mp4 animation creation
    # ani = animation.ArtistAnimation(fig, frames, interval=10, blit=True,  # blit - без потерь
    #                                 repeat_delay=0)  # repeat_delay - автозацикливание (после 100 кадров)

    Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('teapot.mp4', writer)

    # gif animation creation
    ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=0)
    writer = PillowWriter(fps=24)
    # ani.save("teapot.gif", writer=writer)

    plt.show()


# алгоритм Брезенхема
def alg_Br(x0, y0, x1, y1):
    step = 1  # направление увеличения от x0 до x1
    yx = False  # меняли оси или нет

    # корректировка значений
    if abs(x1 - x0) < abs(y1 - y0):  # если наклон резкий -> меняем оси
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        yx = True

    if x0 > x1 and y0 > y1:  # если координаты нулевой точки > первой -> меняем точки местами
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    elif x0 > x1:
        step = -1
    elif y0 > y1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        step = -1

    delta = abs((y1 - y0) / (x1 - x0))
    error = 0.0
    y = y0
    line = []
    for x in range(int(x0), int(x1), step):
        line.append([x, int(y)] if not yx else [int(y), x])
        error += delta
        if error >= 0.5:
            y += 1 * (1 if (y1 - y0) > 0 else (-1))
            error -= 1

    return tuple(line)


if __name__ == '__main__':
    vertex, face = unpack('teapot.obj')
    vertexxy = change_xy(vertex)
    # image = paint(vertexxy, face)

    frames,fig = make_frames(vertexxy, face)
    create_anim(frames,fig)
