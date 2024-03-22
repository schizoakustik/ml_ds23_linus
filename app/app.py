import joblib
from requests import session
import streamlit as st
from streamlit_cropper import st_cropper
from streamlit_super_slider import st_slider
from PIL import Image
import pandas as pd
import numpy as np
import skimage as ski
from src.model import preprocess, predict_sudoku, plot_solution, plot_image, split_boxes
from src.solver import SudokuSolver

clf = joblib.load('svc.pkl')
nbin_values = [2**i for i in range(1, 11)]

def add_su_df():
    st.session_state['su_df'] = su_df

def update_su_df():
    state = st.session_state['su_df_edit']
    for index, updates in state['edited_rows'].items():
        for key, value in updates.items():
            st.session_state['su_df'].loc[st.session_state['su_df'].index == int(index), int(key)] = value

def editor():
    # if 'su_df' not in st.session_state:
        # st.session_state['su_df'] = su_df
    # st.session_state['su_df_solved'] = st.session_state['su_df'].copy()
    st.data_editor(st.session_state['su_df'], key='su_df_edit', hide_index=True)

def solve_sudoku():
    update_su_df()
    sudoku = st.session_state['su_df'].values.tolist()
    solver = SudokuSolver(sudoku=sudoku)
    if solver.solve():
        st.session_state['solved'] = True
        st.session_state['su_df_solved'] = solver.as_array()
        st.balloons()
    else:
        st.session_state['solved'] = False
        st.toast('Kunde inte lösa sudokut...', icon='😔')

def show_solution():
    st.pyplot(fig=plot_solution(np.array(cropped)/255, st.session_state['su_df_solved']))

if 'solved' not in st.session_state:
    st.session_state['solved'] = False

if 'nbins' not in st.session_state:
    st.session_state['nbins'] = 256

if 'min_shape_size' not in st.session_state:
    st.session_state['min_shape_size'] = 100

st.title('Sudoku-lösare 3000')
mode = st.toggle(label='Kamera/Filuppladdning', value=True)
if mode:
    image = st.file_uploader('Ladda upp en bild på ett sudoku', type=['png', 'jpg', 'jpeg'])
else:
    image = st.camera_input('Ta en bild av ett sudoku')

st.divider()

if image:
    # st.image(image)
    # img = ski.io.imread(image)
    img = Image.open(image)
    cropped = st_cropper(img, aspect_ratio=(1, 1))
    if 'su_df' not in st.session_state:
        with st.spinner('Arbetar...'):
            img_prep = preprocess(np.array(cropped)/255)
            boxes, su_df = split_boxes(img_prep, clf)
                # st.session_state['su_df'] = su_df
            st.pyplot(boxes)
            st.button('Nästa steg', on_click=add_su_df)
    if 'su_df' in st.session_state:
        st.text('Kolla om det är rätt siffror, ändra de som blivit fel')
        # fig, su_df = predict_sudoku(img_prep, clf, boxes)
        col1, col2 = st.columns((1, 2))

        with col1:
            st.image(img)
            st.caption('Originalbilden')
            # st.pyplot(fig)
        with col2:
            editor()
            st.button('Lös sudoku', type='primary', on_click=solve_sudoku)
            if st.session_state['solved']:
                solution = st.button(label='Visa lösning', on_click=show_solution)
                # st.button(label='Ge mig en hint')