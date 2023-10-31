import pandas as pd
import numpy as np
import sys
from commons import USERS, LB_API_KEY, DATAPATH
from workers import LBWorker


class DialogAnnotation():

    def __init__(self, data_reader=LBWorker(LB_API_KEY, DATAPATH)):

        # data is a list of dialog objects
        self.dialogs = []
        self.data_reader = data_reader

    def add_projects(self, list_of_projects):
        for project in list_of_projects:
            data = self.data_reader.read_data(project, update=True)
            for d in data:
                self.dialogs.append((Dialog(d)))

    def get_dialog_by_name(self, dialog_name):
        return [dialog for dialog in self.dialogs if dialog.name==dialog_name][0]

    def get_dialogs(self):
        return self.dialogs


class Dialog():

    def __init__(self, lb_datarow):

        self.text = lb_datarow['data_row']['row_data']
        self.name = lb_datarow['data_row']['details']['dataset_name'][:-5]
        self.project_id = list(lb_datarow['projects'].keys())[0]
        self.span_id_table = dict()

        self.labels = self.index_spans(
                        list([Label(self, lb_label)
                            for lb_label in
                            list(lb_datarow['projects'].items())[0][1]['labels']
                                if len(lb_label['annotations']['objects']) > 1]))
        self.agreement = Agreement(self)

    def __repr__(self):
        return f"Dialog({self.name}, {self.project_id}, {len(self.labels)})"

    def get_text(self):
        return self.text

    def get_labels(self):
        return self.labels

    def get_labelers(self):
        return [label.get_labeler() for label in self.get_labels()]

    def get_agreement(self):
        return self.agreement

    def get_project_id(self):
        return self.project_id

    def get_name(self):
        return self.name

    def index_spans(self, list_of_labels):
        """Input: list of Label objects,
           Outpu: list of Label objects, with indices fixed"""


        ranges = (pd.concat([label.get_spans() for label in list_of_labels], ignore_index=True)
                  .assign(range = lambda df: list(zip(df.loc[:,'start'], df.loc[:,'end'])))
                  .sort_values(by='range')
                  .drop_duplicates(subset=['range'], keep='first')
                 )

        range_to_id = dict(zip(ranges.loc[:,'range'], ranges.index))

        old_id_to_range = (pd.concat([label.get_spans() for label in list_of_labels], ignore_index=True)
                           .assign(range = lambda df: list(zip(df.loc[:,'start'], df.loc[:,'end'])))
                           .drop_duplicates(subset=['id'], keep='first')
                           .drop(['start','end'], axis=1)
                           .set_index('id')
                            )

        for label in list_of_labels:
            label.set_spans(
                            (label.get_spans() .assign(range = lambda df: list(zip(df.loc[:,'start'], df.loc[:,'end']))) .assign(id = lambda df: df.loc[:,'range'].apply(lambda x: range_to_id[x])) .assign(text = lambda df: df['range'].apply(lambda x: self.text[x[0]:x[1]+1]))
                             .drop('range', axis=1)))

            label.set_relations(
                                 (label.get_relations()
                                  .assign(fro = lambda df: df.loc[:,'from'].apply(lambda x: range_to_id[old_id_to_range.at[x,'range']]))
                                  .assign(to = lambda df: df.loc[:,'to'].apply(lambda x: range_to_id[old_id_to_range.at[x,'range']]))
                                  .drop('from', axis=1)
                                  .rename({'fro':'from'}, axis=1)
                                  .pipe(lambda df: df.loc[:,['from','to','tag']])
                                 ))
        return list_of_labels


class Label():

    def __init__(self, owner, lb_label):

        self.labeler = USERS[lb_label['label_details']['created_by']]
        self.spans = self.tabulate_spans(lb_label['annotations']['objects'])
        self.relations = self.tabulate_relations(lb_label['annotations']['relationships'])

    def tabulate_spans(self, lb_spans):
        spans = [{'id': span['feature_id'],
                  'start': span['location']['start'],
                   'end': span['location']['end']}
                for span in lb_spans]

        return pd.DataFrame(spans, columns=['id','start','end'])

    def tabulate_relations(self, lb_relations):
        relations = [{'tag': relation['name'],
                      'from': relation['unidirectional_relationship']['source'],
                      'to': relation['unidirectional_relationship']['target']}
                    for relation in lb_relations]

        return pd.DataFrame(relations, columns=['tag','from','to'])

    def get_spans(self):
        return self.spans
    def get_relations(self):
        return self.relations
    def get_labeler(self):
        return self.labeler
    def set_spans(self, df):
        self.spans = df
    def set_relations(self, df):
        self.relations = df
    def __str__(self):
        return f'Spans:\n\n{self.get_spans().to_string()}\n\nRelations:\n\n{self.get_relations().to_string()}'

class Agreement():

    def __init__(self, dialog):

        self.left, self.right = dialog.get_labels()
        self.labelers = self.left.get_labeler(), self.right.get_labeler()

        self.matched_spans = self._compute_matched_spans()
        self.unmatched_spans = self._compute_unmatched_spans()
        self.disagreed_relations = self._compute_disagreed_relations()
        self.unmatched_relations = self._compute_unmatched_relations()

    def _compute_disagreed_relations(self):
        return (self.left.get_relations().merge(self.right.get_relations(),
                                                on=['from','to'],
                                                how='inner',
                                                suffixes=[f'_{l}' for l in self.labelers])
                .query(f'tag_{self.labelers[0]} != tag_{self.labelers[1]}')
                .sort_values(by="from")
                .reset_index()
               )

    def _compute_unmatched_relations(self):
        return (self.left.get_relations().merge(self.right.get_relations(),
                                                on=['from','to'],
                                                how='outer',
                                                indicator=True,
                                                suffixes=[f'_{l}' for l in self.labelers])
                .query('_merge != "both"')
                .assign(labeler = lambda df: df.apply(
                    lambda df: self.labelers[0] if pd.isnull(df[f'tag_{self.labelers[1]}']) else self.labelers[1],
                    axis='columns'))
                .assign(tag = lambda df: df.apply(
                    lambda df: df[f'tag_{self.labelers[0]}'] if pd.isnull(df[f'tag_{self.labelers[1]}']) else df[f'tag_{self.labelers[1]}'],
                    axis='columns'))
                .pipe(lambda df: df.drop(columns=[column for column in df.columns if '_' in column]))
                .reset_index()
               )

    def _compute_matched_spans(self):
        return (self.left.get_spans().merge(self.right.get_spans(),
                                           how='inner')
                .sort_values(by="id")
                .reset_index()
               )

    def _compute_unmatched_spans(self):


        return (self.left.get_spans().merge(self.right.get_spans(),
                                            how='outer',
                                            indicator=True,
                                            on='id',
                                            suffixes=[f'_{l}' for l in self.labelers])
                .sort_values(by="id")
                .query('_merge != "both"')
                .drop(columns=['_merge'])
                .assign(start = lambda df: df.apply(
                    lambda df: df[f'start_{self.labelers[0]}'] if np.isnan(df[f'start_{self.labelers[1]}']) else df[f'start_{self.labelers[1]}'],
                    axis='columns'))
                .assign(end = lambda df: df.apply(
                    lambda df: df[f'end_{self.labelers[0]}'] if np.isnan(df[f'end_{self.labelers[1]}']) else df[f'end_{self.labelers[1]}'],
                    axis='columns'))
                .assign(text = lambda df: df.apply(
                    lambda df: df[f'text_{self.labelers[0]}'] if  pd.isnull(df[f'text_{self.labelers[1]}'])  else df[f'text_{self.labelers[1]}'],
                    axis='columns'))
                .assign(labeler = lambda df: df.apply(
                    lambda df: self.labelers[0] if np.isnan(df[f'start_{self.labelers[1]}']) else self.labelers[1],
                    axis='columns'))
                .pipe(lambda df: df.drop(columns = [column for column in df.columns if '_' in column]) )
                .reset_index()
               )

    def get_matched_spans(self):
        return self.matched_spans

    def get_unmatched_spans(self):
        return self.unmatched_spans

    def get_disagreed_relations(self):
        return self.disagreed_relations

    def get_unmatched_relations(self):
        return self.unmatched_relations
