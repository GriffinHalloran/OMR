#!/usr/local/bin/python3

from PIL import Image, ImageFilter, ImageDraw, ImageFont
import random
import math
from copy import deepcopy
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import sys

flag = False

def load_image(path):
    im = Image.open(path).convert("L")
    return im

def pad_image(I, H):
    k = len(H)//2
    padded_I = Image.new(I.mode, (I.width+k, I.height+k), color=0)

    for i in range(0, I.width):
        for j in range(0, I.height):
            padded_I.putpixel((i+k, j+k), I.getpixel((i, j)))

    for i in range(padded_I.width):
        val = 2*k-1
        for j in range(k):
            padded_I.putpixel((i, j), padded_I.getpixel((i, val)))
            val -= 1

    for i in range(padded_I.width):
        val = padded_I.height-1-k
        for j in range(padded_I.height-k-1, padded_I.height):
            padded_I.putpixel((i, j), padded_I.getpixel((i, val)))
            val -= 1

    for j in range(padded_I.height):
        val = 2*k-1
        for i in range(k):
            padded_I.putpixel((i, j), padded_I.getpixel((val, j)))
            val -= 1

    for j in range(padded_I.height):
        val = padded_I.width-1-k
        for i in range(padded_I.width-k-1, padded_I.width):
            padded_I.putpixel((i, j), padded_I.getpixel((val, j)))
            val -= 1

    return padded_I

# Create box filter kernel
def create_2d_kernel(kernel_size):
    H = [[1/9 for i in range(kernel_size)] for j in range(kernel_size)]
    return H

def convolve_2d(I, H):
    kr = len(H)//2
    kc = len(H[0])//2
    
    I_copy = I.copy()
    for i in range(kc, I.width-kc):
        for j in range(kr, I.height-kr):
            sum_u = 0
            for u in range(-kc, kc+1):
                sum_v = 0
                for v in range(-kr, kr+1):
                    if (u+kr) < len(H) and (v+kc) < len(H[0]):
                        new_val = H[u+kr][v+kc] * I.getpixel((i-u, j-v))
                        sum_v += new_val
                sum_u += sum_v
            I_copy.putpixel((i, j), int(sum_u))

    return I_copy

def create_separable_kernels(kernel_size):
    Hx = [1/3 for i in range(kernel_size)]
    Hy = [[1/3] for i in range(kernel_size)]
    return Hx, Hy

def convolve_separable(I, Hx, Hy):
    I_copy = I.copy()
    k = len(Hx)//2
    for i in range(k, I.width-k):
        for j in range(k, I.height-k):
            sum_u = 0
            for u in range(-k, k+1):
                if (u+k) < len(Hx):
                    new_val = Hx[u+k] * I.getpixel((i-u, j))
            sum_u += new_val
        I_copy.putpixel((i, j), int(sum_u))

    for i in range(k, I.width-k):
        for j in range(k, I.height-k):
            sum_v = 0
            for v in range(-k, k+1):
                if (v+k) < len(Hy):
                    new_val = Hy[v+k][0] * I.getpixel((i, j-v))
            sum_v += new_val
        I_copy.putpixel((i, j), int(sum_v))
    return I_copy

def create_sobel_operator():
    Hx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    Hy = [[1, 2, 1],
          [0, 0, 0],
          [-1, -2, -1]]

    return Hx, Hy

def edge_detector(I, Sx, Sy):
    I_copy = I.copy()
    res1 = convolve_2d(I, Sx)
    res2 = convolve_2d(I_copy, Sy)
    res3 = I.copy()
    for i in range(res1.width):
        for j in range(res1.height):
            pix1 = res1.getpixel((i, j))**2
            pix2 = res2.getpixel((i, j))**2

            pix_sum = min(255, math.sqrt(pix1 + pix2))

            res3.putpixel((i, j), int(pix_sum))
    
    enhanced = invert_edge_image(res3)

    return enhanced

def get_binary_image(I):
    result = [[0 for i in range(I.height)] for j in range(I.width)]
    for i in range(0, I.width):
        for j in range(0, I.height):
            if I.getpixel((i, j)) >= 128:
                result[i][j] = 1
            else:
                result[i][j] = 0
    return result

def get_inverse_binary_image(I):
    result = [[0 for i in range(I.height)] for j in range(I.width)]
    for i in range(0, I.width):
        for j in range(0, I.height):
            if I.getpixel((i, j)) >= 128:
                result[i][j] = 0
            else:
                result[i][j] = 1
    return result

def invert_edge_image(I):
    result = [[0 for i in range(I.height)] for j in range(I.width)]
    for i in range(0, I.width):
        for j in range(0, I.height):
            if I.getpixel((i, j)) >= 128:
                result[i][j] = 255
            else:
                result[i][j] = 0
            
    return get_image(result)

def save_image(arr, filename):
    arr = np.array(arr)
    arr = np.transpose(arr)
    Image.fromarray(arr.astype(np.uint8)).save(filename, "PNG")

def get_image(arr):
    arr = np.array(arr)
    arr = np.transpose(arr)
    return Image.fromarray(arr.astype(np.uint8))

def normalize_image(I):
    maximum = 0.0
    for i in range(0, I.width):
        for j in range(0, I.height):
            maximum = max(maximum, I.getpixel((i, j)))

    for k in range(0, I.width):
        for l in range(0, I.height):
            pixel_value = I.getpixel((k, l))
            pixel_value = pixel_value / maximum
            pixel_value = pixel_value * 255.0
            I.putpixel((k, l), int(pixel_value))
    
    return I

def gamma(val):
    if val == 0:
        return math.inf
    else:
        return 0


def calculate_D(I):
    print("Calculating D") 
    D = [[0 for i in range(len(I))] for j in range(len(I[0]))]

    for i in range(len(D[0])):
        for j in range(len(D)):
            
            min_val = math.inf
            for a in range(len(D[0])):
                for b in range(len(D)):

                    val = gamma(I[a][b]) + math.sqrt((i-a)**2 + (j-b)**2)
                    min_val = min(min_val, val)
                    
            D[i][j] = min_val
            
    print("Calculated D")
    return D

def flip_kernel(H):
    H_h = [[0 for i in range(len(H))] for j in range(len(H[0]))]
    for i in range(len(H)):
        for j in range(len(H[0])):
            H_h[i][j] = H[i][len(H[0])-j-1]

    H_v = [[0 for i in range(len(H))] for j in range(len(H[0]))]

    for i in range(len(H[0])):
        for j in range(len(H)):
            H_v[i][j] = H_h[len(H[0])-i-1][j]
    return H_v

def convolve_with_template(I, template):
    result = [[0 for i in range(I.height)] for j in range(I.width)]
    for i in range(I.width):
        for j in range(I.height):
            result[i][j] = 0
            for k in range(template.width):
                for l in range(template.height):
                    if (i-k+1 >= 0 and (i-k+1) < I.width and j-l+1 >= 0 and (j-l+1) < I.height):
                        result[i][j] += I.getpixel((i-k+1,j-l+1)) * template.getpixel((k,l))
                    
    return result

def inverse_x_y(I):
    result = [[0 for i in range(I.height)] for j in range(I.width)]
    for i in range(0, I.width):
        for j in range(0, I.height):
            result[i][j] = I.getpixel((I.width - 1 - i, I.height - 1 - j))
    return result

def template_matching_1(I, template, sym, threshold, bottom=0, dist=0, treble=False, detect_pitch=True):

    symbols = []
    binary_image = get_binary_image(I)
    save_image(binary_image, "binary.png")
    inverse_binary_image = get_inverse_binary_image(I)
    save_image(inverse_binary_image, "inverse_binary.png")
    template_binary_image = get_binary_image(template)
    inverse_template_binary_image = get_inverse_binary_image(template)

    template_I = get_image(template_binary_image)
    inverse_binary_I = get_image(inverse_binary_image)
    inverse_template_I = get_image(inverse_template_binary_image)
    template_binary_image_x_y = inverse_x_y(template_I)
    inverse_template_binary_image_x_y = inverse_x_y(inverse_template_I)

    binary_I = get_image(binary_image)
    template_I = get_image(template_binary_image_x_y)
    inverse_template_I = get_image(inverse_template_binary_image_x_y)
    output = convolve_with_template(binary_I, template_I)
    output_inverse = convolve_with_template(inverse_binary_I, inverse_template_I)
    normalized_output = normalize_image(get_image(output))
    normalized_output_inverse = normalize_image(get_image(output_inverse))
    normalized_output.save("normalized_output.png", "PNG")
    normalized_output_inverse.save("normalized_output_inverse.png", "PNG")

    output = get_image(normalized_output)
    output_inverse = get_image(normalized_output_inverse)

    distance = [[0 for i in range(output.height)] for j in range(output.width)]

    template1_symbols = []

    for i in range(0, output.width):
        for j in range(0, output.height):
            distance[i][j] = output.getpixel((i,j)) + output_inverse.getpixel((i,j))
    
    distance_I = get_image(distance)
    normalized_distance = normalize_image(distance_I)
    normalized_distance_I = get_image(normalized_distance)


    for k in range(0, normalized_distance_I.width):
        for l in range(0, normalized_distance_I.height):
            if normalized_distance_I.getpixel((k, l)) > threshold:
                global flag 
                flag = False

                for i in range(int(-template.width/2), int(template.width/2)):
                    for j in range(int(-template.height/2), int(template.height/2)):
            
                        try:
                            if normalized_distance_I.getpixel((k+i, l+j)) > normalized_distance_I.getpixel((k, l)) and flag==False:
                                flag = True
                        except Exception as e:
                            pass

                if flag==False:
                    symbols.append({
                        'row': k,
                        'col': l,
                        'template_width': template.width,
                        'template_height': template.height,
                        'symbol': sym,
                        'pitch': '0'
                    })
    
    return symbols

def detect_template_method_1(I, templates):
    bottom = 198
    dist = 10
    symbols1 = template_matching_1(I, templates[0], 'NOTEHEAD', 228, bottom=bottom, dist=dist, treble=False, detect_pitch=True)
    bottom = 78
    symbols1_treble = template_matching_1(I, templates[0], 'NOTEHEAD', 228, bottom=bottom, dist=dist, treble=True, detect_pitch=True)
    symbols2 = template_matching_1(I, templates[1], 'QUARTERREST', 250, detect_pitch=False)
    symbols3 = template_matching_1(I, templates[2], 'EIGHTHREST', 250, detect_pitch=False)


def detect_template_method_2(I, T1, T2, T3, lines, size):
    Hx, Hy = create_sobel_operator()
    I_edge = edge_detector(I, Hx, Hy)
    T1_edge = edge_detector(T1, Hx, Hy)
    T2_edge = edge_detector(T2, Hx, Hy)
    T3_edge = edge_detector(T3, Hx, Hy)

    I_bin = get_binary_image(I_edge)
    T1_bin = get_binary_image(T1_edge)
    T2_bin = get_binary_image(T2_edge)
    T3_bin = get_binary_image(T3_edge)

    #D = calculate_D(I_bin)

    D_mock = [[random.randint(0, 255) for i in range(len(I_bin))] for j in range(len(I_bin[0]))]

    D_im = get_image(D_mock)
    
    F1 = convolve_2d(D_im, T1_bin)
    F2 = convolve_2d(D_im, T2_bin)
    F3 = convolve_2d(D_im, T3_bin)

    # Detect notes from F
    maximum_val1 = 0.0
    for i in range(0, F1.width):
        for j in range(0, F1.height):
            maximum_val1 = max(maximum_val1, F1.getpixel((i, j)))
    threshold1 = maximum_val1 // 2

    maximum_val2 = 0.0
    for i in range(0, F2.width):
        for j in range(0, F2.height):
            maximum_val2 = max(maximum_val2, F2.getpixel((i, j)))
    threshold2 = maximum_val2 // 2

    maximum_val3 = 0.0
    for i in range(0, F3.width):
        for j in range(0, F3.height):
            maximum_val3 = max(maximum_val3, F3.getpixel((i, j)))
    threshold3 = maximum_val3 // 2

    T1_symbols = []
    T2_symbols = []
    T3_symbols = []

    # Template 1
    kr1 = len(T1_bin)//2
    kc1 = len(T1_bin[0])//2
    
    for i in range(kc1, F1.width-kc1):
        for j in range(kr1, F1.height-kr1):
            val = F1.getpixel((i, j))
            if val >= threshold1:
                T1_symbols.append((i, j, val**2))
    # Template 2
    kr2 = len(T2_bin)//2
    kc2 = len(T2_bin[0])//2
    
    for i in range(kc2, F2.width-kc2):
        for j in range(kr2, F2.height-kr2):
            val = F2.getpixel((i, j))
            if val >= threshold2:
                T2_symbols.append((i, j, val**2))
    # Template 3
    kr3 = len(T3_bin)//2
    kc3 = len(T3_bin[0])//2
    
    for i in range(kc3, F3.width-kc3):
        for j in range(kr3, F3.height-kr3):
            val = F3.getpixel((i, j))
            if val >= threshold3:
                T3_symbols.append((i, j, val**2))


    # Draw notes into image

    I_drawable = ImageDraw.Draw(I)

    # Template 1
    for (i, j, _) in T1_symbols:
        x0 = i - kc1
        y0 = j - kr1
        x1 = i + kc1
        y1 = j + kr1
        I_drawable.rectangle([x0, y0, x1, y1], fill = None, outline = 'red')

    # Template 2
    for (i, j, _) in T2_symbols:
        x0 = i - kc2
        y0 = j - kr2
        x1 = i + kc2
        y1 = j + kr2
        I_drawable.rectangle([x0, y0, x1, y1], fill = None, outline = 'green')

    # Template 3
    for (i, j, _) in T3_symbols:
        x0 = i - kc3
        y0 = j - kr3
        x1 = i + kc3
        y1 = j + kr3
        I_drawable.rectangle([x0, y0, x1, y1], fill = None, outline = 'blue')


    I.save("detected.png")

    # Write notes to file

    f = open("detected.txt", 'w+')

    for (i, j, c) in T1_symbols:
        pitch = (math.inf, 0)

        for key, value in lines.items():
            pitch = (min(pitch[0], abs(value - i)), key)

        line = "" + str(i-kc1) + " " + str(j-kr1) + " " + str(kc1*2) + " " + str(kr1*2) + " filled_note " + pitch[1] + " " + str(c) + "\n"
        f.write(line)
    
    for (i, j, c) in T2_symbols:
        line = "" + str(i-kc2) + " " + str(j-kr2) + " " + str(kc2*2) + " " + str(kr2*2) + " quarter_rest _ " + str(c) + "\n"
        f.write(line)
    
    for (i, j, c) in T3_symbols:
        line = "" + str(i-kc3) + " " + str(j-kr3) + " " + str(kc3*2) + " " + str(kr3*2) + " eighth_rest _ " + str(c) + "\n"
        f.write(line)

    f.close()


def make_accumulator(I):
    accumulator = I
    final = I
    height, width = I.shape
    for i in range(0, height):
        for j in range(0, width):
            if 175 <= I[i, j]:
                for q in range(i + 5, height):
                    if(175 <= I[i, j]):
                        accumulator[i][q - i] += 1

    for i in range(0, height):
        for j in range(0, width):
            if I[i][j] > 175 and accumulator[i][j] > 400:
                for q in range(0, width):
                    final[i + j][q] = 255
    return final


def find_staff_lines(path):
    img = cv2.imread(path)
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    Hx, Hy = create_sobel_operator()
    edges = edge_detector(img, Hx, Hy)
    
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    
def staff(path):
    final = cv2.imread(path)
    test = cv2.imread(path, 0)
    rows, cols = test.shape
    pixel_num = []

    for i in range(rows):
        cur_row = test[i]
        num = 0
        for p in range(len(cur_row)):
            if(cur_row[p] < 170):
                num += 1

        pixel_num.append((num, i))

    pixel_num = sorted(pixel_num, reverse = True)
    lines = []
    for i in range(0, len(pixel_num)):
        test = True
        if (pixel_num[i][1] not in lines):
            if len(lines) > 10:
                break
            if len(lines) == 0:
                lines.append(pixel_num[i][1])
            else:
                for p in lines:
                    if abs(p - pixel_num[i][1]) < 3:
                        test = False
                if test == True:
                    lines.append(pixel_num[i][1])


    for i in range(0, 10):
        cv2.line(final, (0, lines[i]), (cols, lines[i]), (0, 0,255), 1, cv2.LINE_AA)
    size = abs(lines[0] - lines[1])
    lines = sorted(lines)
    staff_lines = {}
    staff_lines['G'] = lines[9]
    staff_lines['B'] = lines[8]
    staff_lines['D'] = lines[7]
    staff_lines['F'] = lines[6]
    staff_lines['A'] = lines[5]

    staff_lines['E_treble'] = lines[4]
    staff_lines['G_treble'] = lines[3]
    staff_lines['B_treble'] = lines[2]
    staff_lines['D_treble'] = lines[1]
    staff_lines['F_treble'] = lines[0]

    return staff_lines, size

def resize(img, size):
    #Code from tutorialkart.com for image resizing 
    scale = 11 / size 
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] *scale)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
    cv2.imwrite('test-images/resized.png', resized)
    return 'test-images/resized.png'

if __name__ == "__main__":

    # STEP 1: Load image(given as cmd argument) and templates and pad them.
    if len(sys.argv) != 2:
        raise Exception("Error: Insufficient arguments")

    file_name = sys.argv[1]

    t1_path = "test-images/template1.png"
    t2_path = "test-images/template2.png"
    t3_path = "test-images/template3.png"

    H = create_2d_kernel(3)
    
    image = pad_image(load_image(file_name), H) 

    template1 = pad_image(load_image(t1_path), H)
    template2 = pad_image(load_image(t2_path), H)
    template3 = pad_image(load_image(t3_path), H)

    # STEP 2: Detect Lines and Rescale image
    img = cv2.imread(file_name)
    lines, size = staff(file_name)
    resized = resize(img, size)
    I_resized = load_image(resized)
    I_padded = pad_image(I_resized, H)

    # STEP 3: Detect notes

    #padded_templates = [template1, template2, template3]
    #detect_template_method_1(I_padded, padded_templates)

    detect_template_method_2(I_padded, template1, template2, template3, lines, size)
