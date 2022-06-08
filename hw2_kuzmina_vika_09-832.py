import numpy as np
import matplotlib.pyplot as plt

vertex = []
face = []

N = 1024


def unpack(file):
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


def change_xy():
    vertexxy = np.array(vertex)
    vertexxy = vertexxy[:, 0:2]
    max_x = max(vertex[i][0] for i in range(len(vertex)))
    max_y = max(vertex[i][1] for i in range(len(vertex)))
    mmax = max(max_x, max_y)

    vertexxy[:, 0] /= mmax
    vertexxy[:, 1] /= mmax
    vertexxy[:, 0:2] *= (N / 2 - 1)
    vertexxy[:, 0:2] += N / 2  # сдвиг в центр, а новые координаты центра (512, 512)

    # переворот чайника
    r = 0
    for ver in vertexxy:
        r = ver[1]-512
        ver[1] -= r*2

    return vertexxy


def paint(vertexxy):
    img = np.zeros((N, N, 3), dtype=np.uint8)
    color = np.array([200, 0, 80], dtype=np.uint8)
    plt.figure()
    k = 0
    for trian in face:
        # прорисовываем все 3 ребра
        line1 = alg_Br(vertexxy[trian[0] - 1][0], vertexxy[trian[0] - 1][1],
                       vertexxy[trian[1] - 1][0], vertexxy[trian[1] - 1][1])
        line2 = alg_Br(vertexxy[trian[1] - 1][0], vertexxy[trian[1] - 1][1],
                       vertexxy[trian[2] - 1][0], vertexxy[trian[2] - 1][1])
        line3 = alg_Br(vertexxy[trian[0] - 1][0], vertexxy[trian[0] - 1][1],
                       vertexxy[trian[2] - 1][0], vertexxy[trian[2] - 1][1])
        # if line1 == ():
        #     k += 1
        # if line2 == ():
        #     k += 1
        # if line3 == ():
        #     k += 1

        # !img[номер строки (по y), номер столбца (по x)]
        for pixel in line1:
            img[pixel[1], pixel[0]] = color
        for pixel in line2:
            img[pixel[1], pixel[0]] = color
        for pixel in line3:
            img[pixel[1], pixel[0]] = color

    # print(k, len(face) * 3)  # сравнение кол-ва ребер, которые не рисуются и должны быть нарисованы
    plt.imshow(img) #origin='lower')  # меняю точку начала координат -> поворот изображения
    plt.show()

    return img


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
    unpack('teapot.obj')
    vertexxy = change_xy()
    image = paint(vertexxy)

    # plt.imsave('my_teapot.png',image,origin='lower')

