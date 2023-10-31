import sys
import os
import re
import json
import pandas as pd
import streamlit as st

from commons import LB_PROJECTS
from entities import DialogAnnotation

def display_df(caption, df):
    st.markdown(caption)
    st.dataframe(df)

def display_dialog():

    dialog = da.get_dialog_by_name(st.session_state.dialog_name)
    agreement = dialog.get_agreement()

    st.markdown(f"""
Dialog: {dialog.get_name()}  
Labelers: {', '.join(dialog.get_labelers())}  
""")

    display_df("Matched spans:", agreement.get_matched_spans())
    display_df("Unmatched spans:", agreement.get_unmatched_spans())
    display_df("Disagreed relations:", agreement.get_disagreed_relations())
    display_df("Unmatched relations:", agreement.get_unmatched_relations())

if __name__ == '__main__':

    da = DialogAnnotation()
    da.add_projects(LB_PROJECTS)
    dialogs = da.get_dialogs()


    st.set_page_config(page_title="Dialog Annotation",
                       layout="wide",
                       initial_sidebar_state="expanded")

    st.session_state.dialog_names = [dialog.get_name() for dialog in dialogs]

    st.sidebar.title('Dialogs')
    st.sidebar.selectbox("Select a dialog",
                     key="dialog_name",
                     index=0,
                     options=pd.Series([''] + st.session_state.dialog_names),
                     on_change=display_dialog)

