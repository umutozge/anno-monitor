import sys
import os
import re
import json
import pandas as pd
import streamlit as st

from commons import LB_PROJECTS
from entities import DialogAnnotation

import logging
logging.basicConfig(filename='logfile.log', encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)


class Monitor():

    def __init__(self, dialog_annotation):

        self.da = dialog_annotation

        st.set_page_config(page_title="Dialog Annotation",
                       layout="wide",
                       initial_sidebar_state="expanded")

        st.session_state.dialog_names = [dialog.get_name() for dialog in self.da.get_dialogs()['object'].to_list()]

        st.sidebar.title('Dialogs')
        st.sidebar.selectbox("Select a dialog",
                     key="dialog_name",
                     index=0,
                     options=pd.Series([''] + st.session_state.dialog_names),
                     on_change=self.display_dialog)


        st.sidebar.divider()
        st.sidebar.button("update",
                      help="update the last viewed project",
                      type="primary",
                      on_click=self.update_dialog)

        self.col1, self.col2 = st.columns(2)

    def display_df(self, caption, df):

        st.markdown(caption(df))
        st.dataframe(df, hide_index=True)

    def update_dialog(self):

        dialog_name = st.session_state.dialog_name
        if dialog_name:
            dialog = self.da.get_dialog('name', dialog_name)
            self.da.update(dialog)
            self.display_dialog()
        else:
            st.warning('Please select a dialog.', icon='😄')

    def display_dialog(self):


        dialog_name = st.session_state.dialog_name
        if dialog_name:
            dialog = self.da.get_dialog('name', dialog_name)
            agreement = dialog.get_agreement()

            st.markdown(
                f"""
                Dialog: {dialog.get_name()}  
                Labelers: {', '.join(dialog.get_labelers())}  
                """)

            col1, col2 = st.columns(2)
            with col1:
                self.display_df(lambda x: f"Matched spans ({len(x)}):", agreement.get_matched_spans())
                self.display_df(lambda x: f"Disagreed relations ({len(x)}):", agreement.get_disagreed_relations())

            with col2:
                self.display_df(lambda x: f"Unmatched spans ({len(x)}):", agreement.get_unmatched_spans())
                self.display_df(lambda x: f"Unmatched relations ({len(x)}):", agreement.get_unmatched_relations())

            st.markdown("Text:  ")
            st.markdown(dialog.get_indexed_text(), unsafe_allow_html=True)

if __name__ == '__main__':

    logger.info('\n\n***A fresh start!***\n\n')
    Monitor(DialogAnnotation())