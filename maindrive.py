import os
import re
import json
import pandas as pd
import streamlit as st


from commons import LB_PROJECTS
from entities import DialogAnnotation

import logging
logging.basicConfig(
    filename='logfile.log',
    encoding='utf-8',
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Monitor():

    def __init__(self, dialog_annotation):

        self.da = dialog_annotation

        st.set_page_config(page_title="Dialog Annotation",
                       layout="wide",
                       initial_sidebar_state="expanded")

        st.session_state.dialog_names = sorted([dialog.name for dialog in self.da.dialogs['object'].to_list()])

        st.sidebar.title('DEVELOPMENT VERSION')
        st.sidebar.title('Dialogs')
        st.sidebar.selectbox("Select a dialog",
                     key="dialog_name",
                     index=0,
                     options=pd.Series([''] + st.session_state.dialog_names)
                     )


        st.sidebar.divider()

        st.sidebar.button("view annotation",
                      help="annotations of the last viewed dialog",
                      on_click=self.display_annotation)
        st.sidebar.button("view analysis",
                      help="analysis of the last viewed dialog",
                      on_click=self.display_analysis)
        st.sidebar.button("update",
                      help="update the last viewed dialog",
                      on_click=self.update_dialog)


    def display_df(self, caption, df, hide_index=True):

        st.markdown(caption(df))
        st.dataframe(df, hide_index=hide_index)

    def update_dialog(self):

        dialog_name = st.session_state.dialog_name
        if dialog_name:
            dialog = self.da.get_dialog('name', dialog_name)
            self.da.update(dialog)
            self.display_annotation()
        else:
            st.warning('Please select a dialog.', icon='😄')

    def display_analysis(self):

        dialog_name = st.session_state.dialog_name
        if dialog_name:
            dialog = self.da.get_dialog('name', dialog_name)


            self.display_df(lambda x:'Sentences:', dialog.entity_grid['sentences'])
            self.display_df(lambda x:'Mentions:', dialog.entity_grid['mentions'])
            self.display_df(lambda x:'Links:', dialog.entity_grid['links'])



    def display_annotation(self):

        dialog_name = st.session_state.dialog_name
        if dialog_name:
            dialog = self.da.get_dialog('name', dialog_name)
            agreement = dialog.agreement

            st.markdown(
                f"""
                Dialog: {dialog.name}  
                Labelers: {', '.join(dialog.labelers)}  
                """)

            self.display_df(lambda x: "Summary:",agreement.summary, hide_index=False)

            col1, col2, col3 = st.columns(3)
            with col1:
                self.display_df(lambda x: f"Matched spans ({len(x)}):", agreement.matched_spans)
                self.display_df(lambda x: f"Matched relations ({len(x)}):", agreement.matched_relations)

            with col2:
                self.display_df(lambda x: f"Unmatched spans ({len(x)}):", agreement.unmatched_spans)
                self.display_df(lambda x: f"Unmatched relations ({len(x)}):", agreement.unmatched_relations)

            with col3:
                self.display_df(lambda x: f"Disagreed spans ({len(x)}):", agreement.disagreed_spans)
                self.display_df(lambda x: f"Disagreed relations ({len(x)}):", agreement.disagreed_relations)

            st.markdown("Text:  ")
            st.markdown(dialog.indexed_text, unsafe_allow_html=True)

if __name__ == '__main__':

    logger.info('\n\n***A fresh start!***\n\n')
    Monitor(DialogAnnotation())
