import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial import Delaunay
from pathlib import Path


plane = plt.imread("./plane.png")
red = plt.imread("./red.png")
accept = plt.imread("./accept.png").sum(axis=2) < 4

def extract_anchor_img(x, i):
    sigma = x.sum(axis=2)
    y = np.copy(x)
    red_layer = x[:, :, 0]
    mask = (sigma > 1) & (red_layer > 0.5) | (red_layer < 0.5)
    
    for i in range(3):
        layer = y[:, :, i]
        layer[mask] = 1
        y[:, :, i] = layer
        
    return y

def get_anchor_points(anchor_img):
    m, n, _ = anchor_img.shape
    I = np.tile(np.arange(m).reshape(-1, 1), n)
    J = np.tile(np.arange(n).reshape(-1, 1), m).T
    mask = anchor_img.sum(axis=2) != 3
    return list(zip(I[mask], J[mask]))

def plot_sampled_survivorship(plane, anchor_points, accept, marker, color, size,jitter=20):
    dpi = 80
    h, w, _ = plane.shape
    figsize = w / dpi, h / dpi
    fig = plt.figure(figsize=figsize)
    plt.imshow(plane)
    
    x, y = [], []
    while len(x) < 120:
        i, j = random.choice(anchor_points)
        i = i + np.random.randint(-jitter, jitter+1)
        j = j + np.random.randint(-jitter, jitter+1)

        if accept[i, j]:
            x.append(j)
            y.append(i)

    marker = marker.encode().decode('UTF-8')


    plt.scatter(x, y, s=size, marker=marker, lw=1,c=color)
    plt.gca().set_axis_off()
    st.write(fig)


st.markdown("## Build your own survivorship bias plane!")
st.markdown("#### More on [survivorship bias](https://en.wikipedia.org/wiki/Survivorship_bias#In_the_military) and the famous data science plane.")

st.markdown("Markers from the [Matplotlib API](https://matplotlib.org/stable/api/markers_api.html)")
marker = st.selectbox('Pick a marker!', ('o', 'v', 'X',"D","d","<"), format_func=(lambda x: "Marker:  " + str(x)))
color = st.selectbox('Pick a color!', ('green', 'blue', 'purple','yellow','magenta','orange'), format_func=(lambda x: "Color: " + str(x)))
size = st.selectbox('Pick a dot size!', (100,200,300,400,500))

anchor_img = extract_anchor_img(red, 1)
plt.imshow(anchor_img)
anchor_points = get_anchor_points(anchor_img)
hull = Delaunay(anchor_points)
img = plot_sampled_survivorship(plane, anchor_points, accept,marker,color,size)

st.markdown("### Original Code by [JBN](https://twitter.com/generativist),Riff by [Vicki](http://www.vickiboykis.com)")
st.markdown("[GitHub Repo](https://github.com/veekaybee/plane_lmao)")





