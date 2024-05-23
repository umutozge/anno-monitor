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

    def __init__(self):

        if 'da' in st.session_state:
            self.da = st.session_state.da
        else:
            self.da = DialogAnnotation()
            st.session_state.da = self.da

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

        st.sidebar.button("overview",
                      help="summary of the entire data",
                      on_click=self.display_overview)
        st.sidebar.button("view annotation",
                      help="annotations of the last viewed dialog",
                      on_click=self.display_annotation)
        st.sidebar.button("view analysis",
                      help="analysis of the last viewed dialog",
                      on_click=self.display_analysis)
        st.sidebar.button("sync",
                      help="sync with labelbox",
                      on_click=self.sync)
        st.sidebar.button("dump grids",
                      help="write entity grids to disk",
                      on_click=self.dump_grids)


    def display_df(self, caption, df, hide_index=True):

        st.markdown(caption(df))
        st.dataframe(df, hide_index=hide_index)

    def dump_grids(self):
        self.da.write_grids()

    def sync(self):

        with st.spinner('Updating data, this may take a while...'):
            self.da.update_all()
            self.da = DialogAnnotation()
            st.session_state.da = self.da

    def display_analysis(self):

        dialog_name = st.session_state.dialog_name
        if dialog_name:
            dialog = self.da.get_dialog('name', dialog_name)


            self.display_df(lambda x:'Mentions:', dialog.entity_grid['mentions'])
            self.display_df(lambda x:'Chains:', dialog.entity_grid['chains'])
            st.write(dialog.entity_grid['coref_classes'])
            self.display_df(lambda x:'Links:', dialog.entity_grid['links'])
            self.display_df(lambda x:'Sentences:', dialog.entity_grid['sentences'])

    def display_overview(self):

        self.display_df(lambda x: """Overview:""", self.da.summarize())



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
                self.display_df(lambda x: f"Matched spans ({len(x)}):", agreement.results['matched']['spans'])
                self.display_df(lambda x: f"Matched relations ({len(x)}):", agreement.results['matched']['relations'])

            with col2:
                self.display_df(lambda x: f"Unmatched spans ({len(x)}):", agreement.results['unmatched']['spans'])
                self.display_df(lambda x: f"Unmatched relations ({len(x)}):", agreement.results['unmatched']['relations'])

            with col3:
                self.display_df(lambda x: f"Disagreed spans ({len(x)}):", agreement.results['disagreed']['spans'])
                self.display_df(lambda x: f"Disagreed relations ({len(x)}):", agreement.results['disagreed']['relations'])

            st.markdown("Text:  ")
            st.markdown(dialog.indexed_text, unsafe_allow_html=True)

if __name__ == '__main__':

    logger.info('\n\n***Streamlit restart***\n\n')
    Monitor()
