import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'


def create_anim(frames, fig):
    Writer = animation.writers['ffmpeg']
    ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=0)
    writer = PillowWriter(fps=24)
    # ani.save("numbers.gif", writer=writer)
    # plt.show()


def make_frames(digits):
    frames = []
    fig = plt.figure()
    conversion = np.linspace(0,1,25)

    for i in range(10):  # основные цифры
        begin_num = digits['digit_' + str(i)]
        if i == 9:
            end_num = digits['digit_' + str(0)]
        else:
            end_num = digits['digit_' + str(i + 1)]

        for conv in conversion: # переходные цифры
            # набор сегментов для отрисовки
            segments = []
            for seg in range(4):
                begin_seg = begin_num['segment_' + str(seg)]
                end_seg = end_num['segment_' + str(seg)]
                # набор точек для 1 сегмента
                points = []
                for point in range(4):
                    new_point_x = (1-conv) * begin_seg[point][0] + conv * end_seg[point][0]
                    new_point_y = (1 - conv) * begin_seg[point][1] + conv * end_seg[point][1]
                    points.append([new_point_x,new_point_y])
                segments.append(points)
            # print(segments)
            img = draw_number(segments)
            frames.append([plt.imshow(img)])

    return frames, fig


def draw_number(segments):
    img = np.zeros((500, 500, 3), dtype=np.uint8) + 255
    color = np.array([0, 0, 0], dtype=np.uint8)
    # plt.figure()

    for seg in segments:
        c_points = np.array(seg)
        T = np.linspace(0, 1, 100)
        point_1 = c_points[0]

        for t in T:
            point_2 = alg_deCaste(c_points, t)

            pixels = alg_Br(point_1[0], point_1[1], point_2[0], point_2[1])
            for pixel in pixels:
                img[pixel[1], pixel[0]] = color

            point_1 = [point_2[0], point_2[1]]
    # plt.imshow(img)
    # plt.show()
    return img


# алгоритм деКастольжо
def alg_deCaste(points, t):
    n = len(points)
    for i in range(1, n):
        for j in range(0, n - i):
            points[j] = (1 - t) * points[j] + t * points[j + 1]
    # print(points)
    return points[0]


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
    with open('digits.json', "r") as f:
        digits = json.load(f)

    frames,fig = make_frames(digits)
    create_anim(frames,fig)
