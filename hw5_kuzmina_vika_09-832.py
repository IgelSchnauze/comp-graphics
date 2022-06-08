import math
import random

import numpy as np
import matplotlib.pyplot as plt


def unpack(file):
    vertex = []
    vertex_vt = []
    vertex_vn = []
    face_v = []
    face_vt = []
    face_vn = []
    with open(file, "r") as f:
        for s in f:
            spl_str = s.split()
            if not spl_str:  # pass empty string
                continue

            if spl_str[0] == 'v':
                for i in range(1, len(spl_str)):
                    spl_str[i] = float(spl_str[i])  # type conversion (str->float)
                vertex.append(spl_str[1:])
            elif spl_str[0] == 'vt':
                for i in range(1, len(spl_str)):
                    spl_str[i] = float(spl_str[i])
                vertex_vt.append(spl_str[1:])
            elif spl_str[0] == 'vn':
                for i in range(1, len(spl_str)):
                    spl_str[i] = float(spl_str[i])
                vertex_vn.append(spl_str[1:])
            elif spl_str[0] == 'f':
                # spl_str = '24/1/24' '25/2/25' '26/3/26' = 'v/vt/vn' *3
                len_str = len(spl_str)
                one_face_v = []
                one_face_vt = []
                one_face_vn = []
                for i in range(1, len_str):
                    spl_face = spl_str[i].split('/')
                    for j in range(3):
                        spl_face[j] = int(spl_face[j])  # type conversion  (str->int)
                    one_face_v.append(spl_face[0])
                    one_face_vt.append(spl_face[1])
                    one_face_vn.append(spl_face[2])
                face_v.append(one_face_v)
                face_vt.append(one_face_vt)
                face_vn.append(one_face_vn)
    return np.array(vertex), np.array(vertex_vt), np.array(vertex_vn), \
           np.array(face_v), np.array(face_vt), np.array(face_vn)


def transMatr(vec):
    arr = np.array([[1, 0, 0, vec[0]], [0, 1, 0, vec[1]], [0, 0, 1, vec[2]], [0, 0, 0, 1]])
    return arr


def rotMatr(ang_x, ang_y, ang_z):
    koef_for_ang = np.pi / 180
    ang_x, ang_y, ang_z = ang_x * koef_for_ang, ang_y * koef_for_ang, ang_z * koef_for_ang
    r_x = np.array(
        [[1, 0, 0, 0], [0, np.cos(ang_x), -np.sin(ang_x), 0], [0, np.sin(ang_x), np.cos(ang_x), 0], [0, 0, 0, 1]])
    r_y = np.array(
        [[np.cos(ang_y), 0, np.sin(ang_y), 0], [0, 1, 0, 0], [-np.sin(ang_y), 0, np.cos(ang_y), 0], [0, 0, 0, 1]])
    r_z = np.array(
        [[np.cos(ang_z), -np.sin(ang_z), 0, 0], [np.sin(ang_z), np.cos(ang_z), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return r_x @ r_y @ r_z


def sizeMatr(koef):
    arr = np.array([[koef, 0, 0, 0], [0, koef, 0, 0], [0, 0, koef, 0], [0, 0, 0, 1]])
    return arr


def rotMatr_w2c(vec):
    vec = np.array(vec)
    gamm = vec / np.linalg.norm(vec)
    bett = np.array([0, 1, 0]) - gamm[1] * gamm
    bett = bett / np.linalg.norm(bett)
    alfa = np.cross(bett, gamm)
    alfa = alfa / np.linalg.norm(alfa)

    arr = np.zeros([4, 4])
    arr[:3, 0] = alfa.T
    arr[:3, 1] = bett.T
    arr[:3, 2] = gamm.T
    arr[3, 3] = 1
    return arr


def project_ort(arr_vertex):
    left = min(arr_vertex[0])
    right = max(arr_vertex[0])
    bottom = min(arr_vertex[1])
    top = max(arr_vertex[1])
    near = min(arr_vertex[2])
    far = max(arr_vertex[2])
    max_del = max(right - left, np.abs(top - bottom), np.abs(far - near))
    # середину оставляю, но расстояние до границ делаю одинаковым, чтобы сохранить пропорции
    cen_x = (left + right) / 2
    left = cen_x - max_del / 2
    right = cen_x + max_del / 2
    cen_y = (bottom + top) / 2
    bottom = cen_y - max_del / 2
    top = cen_y + max_del / 2
    cen_z = (near + far) / 2
    near = cen_z - max_del / 2
    far = cen_z + max_del / 2

    arr = np.array([[2 / (right - left), 0, 0, 0], [0, 2 / (top - bottom), 0, 0],
                    [0, 0, -2 / (far - near), 0], [0, 0, 0, 0]])
    arr[:, 3] = [-(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1]
    return arr


def project_per(arr_vertex):
    left = min(arr_vertex[0])
    right = max(arr_vertex[0])
    bottom = min(arr_vertex[1])
    top = max(arr_vertex[1])
    near = min(arr_vertex[2])
    far = max(arr_vertex[2])
    max_del = max(np.abs(right - left), np.abs(top - bottom), np.abs(far - near))
    # середину оставляю, но расстояние до границ делаю одинаковым, чтобы сохранить пропорции
    cen_x = (left + right) / 2
    left = cen_x - max_del / 2
    right = cen_x + max_del / 2
    cen_y = (bottom + top) / 2
    bottom = cen_y - max_del / 2
    top = cen_y + max_del / 2
    cen_z = (near + far) / 2
    near = cen_z - max_del / 2
    far = cen_z + max_del / 2
    arr = np.array([[2 * near / (right - left), 0, 0, 0], [0, 2 * near / (top - bottom), 0, 0],
                    [0, 0, 0, -2 * near * far / (far - near)], [0, 0, 0, 0]])
    arr[:, 2] = [(right + left) / (right - left), (top + bottom) / (top - bottom), -(far + near) / (far - near), -1]
    return arr


def change_xy(arr_vertex, N):
    arr_for_change = arr_vertex.T
    arr_for_change[:, 0:2] *= (N / 2 - 10)  # вычитаю 10, иначе макушка некрасиво упирается в край
    arr_for_change[:, 0:2] += N / 2  # сдвиг в центр (512, 512)

    # переворот
    for ver in arr_for_change:
        r = ver[1] - 512
        ver[1] -= r * 2
    return arr_for_change.T


def to_proj_coords(arr):
    if arr.size == len(arr):  # если массив, а не матрица
        return np.concatenate([arr, [1]])
    r, lenght = arr.shape
    arr = np.concatenate([arr, np.ones((1, lenght))], axis=0)
    return arr


def to_cart_coords(arr):
    arr = arr[:-1] / arr[-1]
    return arr


def new_coord_system(vertex, normal):
    proj_vertex = to_proj_coords(vertex.T)

    Mo2w = transMatr((-1, 0, -2)) @ rotMatr(5, 10, 15) @ sizeMatr(0.8)
    world_vertex = Mo2w @ proj_vertex

    A_cam = (2, 2, 2)
    B_see = (-2, -2, 0)
    B_A = [B_see[i] - A_cam[i] for i in range(3)]
    A_B = [A_cam[i] - B_see[i] for i in range(3)]
    A_neg = [A_cam[i] * (-1) for i in range(3)]
    Mw2c = rotMatr_w2c(A_B).T @ transMatr(A_neg)
    camer_vertex = Mw2c @ world_vertex

    Mproj_ort = project_ort(camer_vertex)
    Mproj_per = project_per(camer_vertex)
    proj_ort_vertex = Mproj_ort @ camer_vertex
    proj_per_vertex = Mproj_per @ camer_vertex  # неправильно считаются!!!
    # print(max(proj_per_vertex[2]) - min(proj_per_vertex[2]))  # y = 2.9, x = 1.84, z = 3.4
    view_vertex = change_xy(proj_ort_vertex, 1024)

    # преобразование нормалей
    inv = lambda arr: np.linalg.inv(arr)
    proj_normal = to_proj_coords(normal.T)
    camer_normal = inv(Mw2c.T) @ inv(Mo2w.T) @ proj_normal
    view_normal_ort = change_xy(inv(Mproj_ort.T) @ camer_normal, 1024)
    view_normal_per = change_xy(inv(Mproj_per.T) @ inv(Mw2c.T) @ inv(Mo2w.T) @ proj_normal, 1024)

    # точка освещения (из мировой к камере)
    light = np.array([-3, 3, 3])
    camera_light = Mw2c @ to_proj_coords(light)

    return to_cart_coords(view_vertex).T, to_cart_coords(view_normal_ort).T, \
           to_cart_coords(camer_vertex).T, to_cart_coords(camer_normal).T, to_cart_coords(camera_light)


def back_face_culling(one_face_norm):
    scalar = np.dot(one_face_norm, [0, 0, -1])  # видно все по направлению Oz
    return scalar


def get_baryc_coords(point, v0, v1, v2):
    T = np.array([v0, v1, v2]).T
    T[2, :] = [1, 1, 1]
    point.append(1)
    V = np.linalg.inv(T) @ point
    return V[0], V[1], V[2]


def lighting(light_point, one_vertex, normal):
    vec_light = (- light_point) - one_vertex
    # фоновое
    k_a = np.array([1, 1, 1])
    i_a = np.array([25, 25, 25])
    l_a = k_a * i_a

    # диффузное
    k_d = np.array([0.8, 1, 0.9])
    i_d = np.array([180, 150, 190])
    cos_d = np.dot(vec_light, normal) / np.linalg.norm(vec_light)  # np.linalg.norm(normal) = 1
    l_d = k_d * i_d * cos_d

    # зеркальное
    k_s = np.array([0.8, 0.8, 0.8])
    i_s = np.array([255, 255, 255])
    glitter = 10
    R = 2 * np.dot(vec_light, normal) * normal - vec_light
    vec_eye = np.array([0, 0, -1])
    # vec_eye = one_vertex
    cos_s = np.dot(R, vec_eye) / (np.linalg.norm(R) * np.linalg.norm(vec_eye))
    l_s = k_s * i_s * (cos_s ** glitter)

    d = np.linalg.norm(vec_light)
    d_a = 1; d_b = 0.02; d_c = 0.004
    koef_d = 1 / (d_a + d_b*d + d_c*(d**2))

    return l_a + koef_d * (l_d + l_s)


def z_buffer_texture(vertexes, faces, vertexes_camera, texture, vertexes_vt, faces_vt,
                     normals_camera, faces_vn, light, need_light):
    N = 1024
    step = 1
    img = np.zeros((N, N, 3), dtype=np.uint8) + 50
    plt.figure()

    max_z = max(vertexes[:, 2])
    buf = np.ones([N, N]) * max_z  # все сначала далеко
    for i, face in enumerate(faces):
        # в-р нормали к грани
        face_normal = np.cross(vertexes_camera[face[1] - 1] - vertexes_camera[face[0] - 1],
                               vertexes_camera[face[2] - 1] - vertexes_camera[face[1] - 1])
        face_normal = face_normal / np.linalg.norm(face_normal)
        scalar = back_face_culling(face_normal)
        if scalar < 0:

            x_min = min([vertexes[face[j] - 1, 0] for j in range(3)])
            x_max = max([vertexes[face[j] - 1, 0] for j in range(3)])
            y_min = min([vertexes[face[j] - 1, 1] for j in range(3)])
            y_max = max([vertexes[face[j] - 1, 1] for j in range(3)])

            for x in range(math.floor(x_min), math.ceil(x_max), step):
                for y in range(math.floor(y_min), math.ceil(y_max), step):
                    a, b, c = get_baryc_coords([x, y], vertexes[face[0] - 1], vertexes[face[1] - 1],
                                               vertexes[face[2] - 1])
                    if a < 0 or b < 0 or c < 0:
                        continue  # идем смотреть след пиксель

                    z = a * vertexes[face[0] - 1, 2] + b * vertexes[face[1] - 1, 2] + c * vertexes[face[2] - 1, 2]
                    if z < buf[x, y]:  # значит рисуем!
                        # получаем текстур коорд-ты вершин грани
                        v0t = vertexes_vt[faces_vt[i, 0] - 1]
                        v1t = vertexes_vt[faces_vt[i, 1] - 1]
                        v2t = vertexes_vt[faces_vt[i, 2] - 1]
                        u, v = a * v0t + b * v1t + c * v2t
                        pixel_textur = texture[1024 - math.ceil(v * N)][math.ceil(u * N)]  # вектор цвета (текстуры)

                        if not need_light:
                            color = np.array(pixel_textur, dtype=np.uint8)
                            img[y, x] = color
                            buf[x, y] = z
                            continue

                        # высчитываем нормаль в пикселе и координаты пикселя в системе камеры
                        pixel_normal = a * np.array(normals_camera[faces_vn[i, 0] - 1]) + \
                                       b * np.array(normals_camera[faces_vn[i, 1] - 1]) + \
                                       c * np.array(normals_camera[faces_vn[i, 2] - 1])
                        pixel_normal = pixel_normal / np.linalg.norm(pixel_normal)
                        cam_x, cam_y, cam_z = a * vertexes_camera[face[0] - 1] + \
                                              b * vertexes_camera[face[1] - 1] + \
                                              c * vertexes_camera[face[2] - 1]
                        pixel_light = lighting(light, np.array([cam_x, cam_y, cam_z]), pixel_normal)  # вектор освещения

                        pixel_textur_light = pixel_textur * (pixel_light / 255)  # текстура + свет
                        pixel_textur_light = np.clip(pixel_textur_light, 0, 255)  # загон значений массива в опр рамки
                        color = np.array(pixel_textur_light, dtype=np.uint8)

                        img[y, x] = color
                        buf[x, y] = z
    plt.imshow(img)
    plt.show()
    return img


def z_buffer_color(vertexes, faces, vertexes_camera):
    N = 1024
    step = 1
    img = np.zeros((N, N, 3), dtype=np.uint8) + 50
    plt.figure()

    max_z = max(vertexes[:, 2])
    buf = np.ones([N, N]) * max_z  # все сначала далеко
    for i, face in enumerate(faces):
        # в-р нормали к грани
        face_normal = np.cross(vertexes_camera[face[1] - 1] - vertexes_camera[face[0] - 1],
                               vertexes_camera[face[2] - 1] - vertexes_camera[face[1] - 1])
        face_normal = face_normal / np.linalg.norm(face_normal)
        scalar = back_face_culling(face_normal)
        if scalar < 0:
            # рандомный цвет
            # color = np.array([random.randint(0,255), random.randint(0,255), random.randint(0,255)], dtype=np.uint8)

            # оттенок серого
            grey_inten = np.abs(scalar * 255)
            if grey_inten > 255:
                grey_inten = 255
            color = np.array([grey_inten, grey_inten, grey_inten], dtype=np.uint8)

            x_min = min([vertexes[face[j] - 1, 0] for j in range(3)])
            x_max = max([vertexes[face[j] - 1, 0] for j in range(3)])
            y_min = min([vertexes[face[j] - 1, 1] for j in range(3)])
            y_max = max([vertexes[face[j] - 1, 1] for j in range(3)])

            for x in range(math.floor(x_min), math.ceil(x_max), step):
                for y in range(math.floor(y_min), math.ceil(y_max), step):
                    a, b, c = get_baryc_coords([x, y], vertexes[face[0] - 1], vertexes[face[1] - 1],
                                               vertexes[face[2] - 1])
                    if a < 0 or b < 0 or c < 0:
                        continue  # идем смотреть след пиксель

                    z = a * vertexes[face[0] - 1, 2] + b * vertexes[face[1] - 1, 2] + c * vertexes[face[2] - 1, 2]
                    if z < buf[x, y]:  # значит рисуем!
                        img[y, x] = color
                        buf[x, y] = z
    plt.imshow(img)
    plt.show()
    return img


def paint_back_cull(vertexes, faces, vertexes_camera):
    N = 1024
    img = np.zeros((N, N, 3), dtype=np.uint8) + 255
    color = np.array([0, 0, 0], dtype=np.uint8)
    plt.figure()

    for i, trian in enumerate(faces):
        face_normal = np.cross(vertexes_camera[trian[1] - 1] - vertexes_camera[trian[0] - 1],
                               vertexes_camera[trian[2] - 1] - vertexes_camera[trian[1] - 1])
        face_normal = face_normal / np.linalg.norm(face_normal)
        if back_face_culling(face_normal) >= 0:
            continue  # пропускаем эту грань

        # прорисовываем все 3 ребра
        line1 = alg_Br(vertexes[trian[0] - 1][0], vertexes[trian[0] - 1][1],
                       vertexes[trian[1] - 1][0], vertexes[trian[1] - 1][1])
        line2 = alg_Br(vertexes[trian[1] - 1][0], vertexes[trian[1] - 1][1],
                       vertexes[trian[2] - 1][0], vertexes[trian[2] - 1][1])
        line3 = alg_Br(vertexes[trian[0] - 1][0], vertexes[trian[0] - 1][1],
                       vertexes[trian[2] - 1][0], vertexes[trian[2] - 1][1])

        # !img[номер строки (по y), номер столбца (по x)]
        for pixel in line1:
            img[pixel[1], pixel[0]] = color
        for pixel in line2:
            img[pixel[1], pixel[0]] = color
        for pixel in line3:
            img[pixel[1], pixel[0]] = color

    plt.imshow(img)
    plt.show()


def paint(vertexes, faces):
    N = 1024
    img = np.zeros((N, N, 3), dtype=np.uint8) + 255
    color = np.array([0, 0, 0], dtype=np.uint8)
    plt.figure()
    for trian in faces:
        # прорисовываем все 3 ребра
        line1 = alg_Br(vertexes[trian[0] - 1][0], vertexes[trian[0] - 1][1],
                       vertexes[trian[1] - 1][0], vertexes[trian[1] - 1][1])
        line2 = alg_Br(vertexes[trian[1] - 1][0], vertexes[trian[1] - 1][1],
                       vertexes[trian[2] - 1][0], vertexes[trian[2] - 1][1])
        line3 = alg_Br(vertexes[trian[0] - 1][0], vertexes[trian[0] - 1][1],
                       vertexes[trian[2] - 1][0], vertexes[trian[2] - 1][1])

        # !img[номер строки (по y), номер столбца (по x)]
        for pixel in line1:
            img[pixel[1], pixel[0]] = color
        for pixel in line2:
            img[pixel[1], pixel[0]] = color
        for pixel in line3:
            img[pixel[1], pixel[0]] = color

    plt.imshow(img)
    plt.show()
    return img


def alg_Br(x0, y0, x1, y1):  # алгоритм Брезенхема
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

    delta = abs((y1 - y0) / (x1 - x0))  # if not x1 - x0 == 0 else 0
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
    # become just np.array
    vertexes, vertexes_vt, vertexes_vn, faces, faces_vt, faces_vn = unpack('obj/african_head/african_head.obj')
    texture = np.array(plt.imread('obj/african_head/african_head_diffuse.tga'))
    vertexes_vt = vertexes_vt[:, :2]

    # преобразование систем координат
    render_vertexes, render_normals, camera_vertexes, camera_normals, camera_light = new_coord_system(vertexes, vertexes_vn)

    img_1 = paint(render_vertexes,faces)
    img_2 = z_buffer_color(render_vertexes, faces, camera_vertexes)
    img_3 = z_buffer_texture(render_vertexes, faces, camera_vertexes, texture, vertexes_vt, faces_vt,
                             None, None, None, False)
    img_4 = z_buffer_texture(render_vertexes, faces, camera_vertexes, texture, vertexes_vt, faces_vt,
                            camera_normals, faces_vn, camera_light, True)

    # plt.imsave('hw5_1_carcass.png', img_1)
    # plt.imsave('hw5_2_intensity.png', img_2)
    # plt.imsave('hw5_3_texture.png', img_3)
    # plt.imsave('hw5_4_light.png', img_4)
