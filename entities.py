import pandas as pd
import numpy as np
import os
import sys
import json
import time

from functools import reduce

from commons import USERS, LB_API_KEY, DATAPATH, LB_PROJECTS, make_color_picker
from workers import LBWorker

import logging
logging.basicConfig(
    filename='logfile.log',
    encoding='utf-8',
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DialogAnnotation:

    def __init__(self, list_of_projects=LB_PROJECTS, data_reader=LBWorker(LB_API_KEY, DATAPATH), datapath=DATAPATH):

        self.dialogs = None
        self.datapath = datapath
        self.data_reader = data_reader
        self.list_of_projects = list_of_projects
        self.load_projects()

    def load_projects(self, update=False):
        store = []
        for project in self.list_of_projects:
            data = self.data_reader.read_data(project, update=update)
            for d in data:
                dialog = Dialog(d, self)
                if dialog:
                    store.append(dialog)

        self.dialogs = (pd.DataFrame({'object': store})
                        .assign(name = lambda df: df['object'].apply(lambda row: row.name))
                        .assign(datarow_id = lambda df: df['object'].apply(lambda row: row.datarow_id))
                        .assign(project_id = lambda df: df['object'].apply(lambda row: row.project_id))
                        )
        return self

    def update(self, dialog):
        self.data_reader.read_data(dialog.project_id, update=True)
        self.load_projects()

    def update_all(self):
        self.laod_projects(update=True)

    def get_dialog(self, key, value):
        return self.dialogs.loc[lambda df: df[key] == value, 'object'].values[0]

    def __len__(self):
        return (len(self.dialogs))

    def __bool__(self):
        return bool(len(self))

class Dialog:

    def __init__(self, lb_datarow, owner):

        self.owner = owner
        self.lb_datarow = lb_datarow
        self.datarow_id=lb_datarow['data_row']['id']
        self.text = lb_datarow['data_row']['row_data']
        self.name = lb_datarow['data_row']['details']['dataset_name'][:-5]
        self.project_id = list(lb_datarow['projects'].keys())[0]
        self.spans = None
        self.relations = None

        self.organize_data()

        if self:
            self.labelers = list(self.spans.loc[:,'labeler'].unique())
            self.agreement = Agreement(self)
            self.indexed_text = self.generate_indexed_text()
            self.write_to_disk()
            self.entity_grid=EntityGrid(self)
            logger.info(f'Formed {self}')

    def __repr__(self):
        return f"Dialog({self.name}, {self.project_id}, {self.datarow_id})"

    def __bool__(self):
        return bool(self.spans is not None)

    def write_to_disk(self):
        with open(os.path.join(self.owner.datapath,f"{self.datarow_id}-{self.name}.json"), 'w', encoding='utf-8') as ouf:
            json.dump(self.lb_datarow, ouf, indent=4, ensure_ascii=False)

    def organize_data(self):
        """Construct dataframes for spans and relations.
        Store in self.spans and self.relations.
        """

        lb_labels = list(self.lb_datarow['projects'].items())[0][1]['labels']

        # check for exactly 2 labels
        if len(lb_labels) != 2:
            return None

        raw_spans = (pd.DataFrame(
                     list(
                         reduce(lambda x,y:x+y,
                                [
                                 [{'id': span['feature_id'],
                                   'start': span['location']['start'],
                                   'end': span['location']['end'] + 1,
                                   'tag': span['name'],
                                   'labeler': USERS[lb_label['label_details']['created_by']]}
                                  for span in lb_label['annotations']['objects']]
                                 for lb_label in lb_labels]))
                                )
                     .assign(range = lambda df: list(zip(df.loc[:,'start'], df.loc[:,'end'])))
                     .sort_values(by=['start','end']))

        if raw_spans.loc[:,'labeler'].unique().size != 2:
            return None

        ranges = (raw_spans
                  .drop_duplicates(subset=['range'], keep='first')
                  .reset_index(drop=True))

        range_to_id = dict(zip(ranges.loc[:,'range'], ranges.index))

        old_id_to_range = (raw_spans
                           .drop_duplicates(subset=['id'], keep='first')
                           .drop(['start','end'], axis=1)
                           .set_index('id'))

        self.spans = (raw_spans
                      .assign(id = lambda df: df.loc[:,'range'].apply(lambda x: range_to_id[x]))
                      .assign(text = lambda df: df['range'].apply(lambda x: self.text[x[0]:x[1]]))
                      .drop('range', axis=1)
                      .reset_index(drop=True))

        self.relations =\
                (pd.DataFrame(
                    list(
                        reduce(lambda x,y:x+y,
                               [
                                   [{'tag': relation['name'],
                                     'from': relation['unidirectional_relationship']['source'],
                                     'to': relation['unidirectional_relationship']['target'],
                                     'labeler': USERS[lb_label['label_details']['created_by']]}
                                    for relation in lb_label['annotations']['relationships']]
                                   for lb_label in lb_labels])),
                    columns=['tag','from','to','labeler'])
                 .assign(fro = lambda df: df.loc[:,'from'].apply(lambda x: range_to_id.get(old_id_to_range.at[x,'range'] if x in old_id_to_range.index else -1, -1)))
                 .assign(to = lambda df: df.loc[:,'to'].apply(lambda x: range_to_id.get(old_id_to_range.at[x,'range'] if x in old_id_to_range.index else -1, -1)))
                 .drop('from', axis=1)
                 .rename({'fro':'from'}, axis=1)
                 .pipe(lambda df: df.loc[:,['from','to','tag','labeler']])
                 .pipe(lambda df: df.loc[~((df['from'] == -1) | (df['to'] == -1)),:])
                )

    def generate_indexed_text(self):

        color = make_color_picker()
        text = self.text
        start_end = (self.spans
                     .drop_duplicates(subset=['id'], keep='first')
                     .assign(tuples =\
                             lambda df:
                             df.apply(lambda row:
                                      [(row['start'],None),(row['end'],row['id'])], axis='columns'))
                     .loc[:,'tuples']
                     .to_list()
                    )

        start_end = list(reduce(lambda x,y:x+y,start_end))
        start_end.sort(key=lambda x:x[0])

        last_edit_point = 0
        accu = ""
        for start_or_end in start_end:
            if start_or_end[1] is not None:
                accu += text[last_edit_point:start_or_end[0]]
                accu += f'<sup>{start_or_end[1]}</sup>]'
                last_edit_point = start_or_end[0]
            else:
                accu += text[last_edit_point:start_or_end[0]]
                accu += f':{color()}['
                last_edit_point = start_or_end[0]


        if last_edit_point < len(text):
            accu += text[last_edit_point:]

        accu = accu.replace('p:','p: ').replace('o:','o: ').replace('\n','\n<br/>')
        return accu

class Agreement:

    def __init__(self, dialog):

        self.spans = dialog.spans
        self.relations = dialog.relations
        self.labelers = dialog.labelers

        self.matched_spans = self._compute_matched_spans()
        self.unmatched_spans = self._compute_unmatched_spans()
        self.disagreed_spans = self._compute_disagreed_spans()
        self.disagreed_relations = self._compute_disagreed_relations()
        self.matched_relations = self._compute_matched_relations()
        self.unmatched_relations = self._compute_unmatched_relations()
        self.summary = self._compute_summary()


    def _compute_summary(self):

        return (pd.DataFrame(
            {'spans':
             {'matched': len(self.matched_spans),
              'unmatched': len(self.unmatched_spans),
              'disagreed': len(self.disagreed_spans)},
             'relations':
             {'matched': len(self.matched_relations),
              'unmatched':len(self.unmatched_relations),
              'disagreed':len(self.disagreed_relations) }})
            .pipe(lambda df: pd.concat([df,pd.DataFrame([df.sum()], index=['total'])]))
            .assign(spans_p = lambda df: df.apply(lambda row: f"{round(row['spans']/df.at['total','spans'] * 100)}%", axis=1),
             relations_p = lambda df: df.apply(lambda row: f"{round(row['relations']/df.at['total','relations'] * 100)}%", axis=1))
            .pipe(lambda df: df.iloc[:,[0,2,1,3]])
        )


    def _compute_matched_relations(self):
        relations = self.relations.groupby('labeler')
        left, right = [relations.get_group(labeler) for labeler in self.labelers]

        return (pd.merge(left,right,
                         on=['from','to','tag'])
                .loc[:,['from','to','tag']]
                .reset_index(drop=True)
               )

    def _compute_unmatched_relations(self):
        relations = self.relations.groupby('labeler')
        left, right = [relations.get_group(labeler) for labeler in self.labelers]

        return (pd.merge(left,right,
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
                .reset_index(drop=True)
               )

    def _compute_disagreed_relations(self):
        relations = self.relations.groupby('labeler')
        left, right = [relations.get_group(labeler) for labeler in self.labelers]

        return (pd.merge(left,right,
                         on=['from','to'],
                         how='inner',
                         suffixes=[f'_{l}' for l in self.labelers])
                .query(f'tag_{self.labelers[0]} != tag_{self.labelers[1]}')
                .sort_values(by="from")
                .pipe(lambda df: df.drop(columns=[column for column in df.columns if column.startswith('labeler')]))
                .reset_index(drop=True)
               )


    def _compute_matched_spans(self):

        labels = self.spans.groupby('labeler')
        left, right = [labels.get_group(labeler) for labeler in self.labelers]

        return (pd.merge(left,right, on=['id','tag'], how='inner')
                .sort_values(by="id")
                .rename(columns={'start_x':'start','end_x':'end','text_x':'text'})
                .pipe(lambda df:
                      df.drop(columns=[column for column in df.columns if column[-2]=='_']))
                .reset_index(drop=True)
               )

    def _compute_unmatched_spans(self):

        labels = self.spans.groupby('labeler')
        left, right = [labels.get_group(labeler) for labeler in self.labelers]

        return (pd.merge(left,right,
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
                .reset_index(drop=True)
               )

    def _compute_disagreed_spans(self):

        labels = self.spans.groupby('labeler')
        left, right = [labels.get_group(labeler) for labeler in self.labelers]

        return (pd.merge(left,right,
                         on='id',
                         how='inner',
                         suffixes=[f'_{l}' for l in self.labelers])
                .query(f'tag_{self.labelers[0]} != tag_{self.labelers[1]}')
                .sort_values(by="id")
                .rename(columns={'start'+self.labelers[0]: 'start', 'end'+self.labelers[0]: 'end', 'text'+self.labelers[0]:'text'})
                .pipe(lambda df: df.drop(columns=[column for column in df.columns if '_' in column and not column.startswith('tag')]))
                .reset_index(drop=True)
               )

class EntityGrid:

    from trtokenizer.tr_tokenizer import SentenceTokenizer
    sentence_tokenizer=SentenceTokenizer()

    def __init__(self, dialog):

        self.dialog = dialog
        self.spans, self.relations = self.create_gold()
        self.text = self.dialog.text
        self.sentences = self.acquire_sentences()
        self.links = self.acquire_links()
        self.mentions = self.acquire_mentions()
        #self.entities = self.acquire_entities()

    def acquire_sentences(self):

        sentences=\
                [Sentence(self, *sent)
                 for sent in
                 list(
                     map(lambda x: (x[0],x[1][0],x[1][1],x[1][2]),
                         enumerate(
                             reduce(lambda result, sent:
                                    result + [(result[-1][1],
                                               self.text.index(sent,result[-1][1])+len(sent),
                                               sent)],
                                    self.sentence_tokenizer.tokenize(self.text),
                                    [(0,0,'')])[1:])))]

        current = 'anchor'
        for sent in sentences:
            if sent.text.startswith('p:'):
                current='presenter'
            elif sent.text.startswith('o:'):
                current='operator'
            sent.set_speaker(current)

        return sentences

    def acquire_links(self):
        return\
                reduce(lambda dic, tup:
                       dic|{tup[0]:[tup[1]]} if tup[0] not in dic.keys()
                       else dic|{tup[0]: dic[tup[0]] + [tup[1]]},
                       [(rel['from'], (rel['to'],rel['tag']))
                        for rel in self.relations.to_dict(orient='index').values()],
                       dict())

    def acquire_mentions(self):
        return\
            (pd.DataFrame(
                [mention.__dict__
                 for mention in
                 [Mention({'id':k}|
                          v|
                          {'owner':self,
                           'role': None,
                           'form': None,
                           'coref': None,})
                  for k, v in self.spans.to_dict(orient='index').items()]
                ],
            )
                .assign(sent = lambda df: df['sentence'].map(lambda x:x.text))
                .drop(columns=['sentence'])
                .rename(columns={'sentence_id':'sent_id'})
                .loc[:,['id','tag','text','sent','sent_id','speaker','role']]
            )


    def get_role(self, id, tag, text):
        link = self.links.get(id)
        return 'SUBJ'

    def acquire_entities(self):
        return None

    def create_gold(self):
        return\
        self.dialog.agreement.matched_spans, self.dialog.agreement.matched_relations

    def __repr__(self):
        return\
                f"EntityGrid({self.dialog.name})"

    def __getitem__(self, field):
        if field=='sentences':
            return pd.DataFrame(
                [sent.__dict__
                 for sent in self.sentences],
                columns=['id','start','end','text','speaker']
            )
        elif field=='mentions':
            return self.mentions
        elif field=='links':
            return\
                    (pd.DataFrame(
                        [{'id': key, 'links':str(val)}
                        for key, val in self.links.items()],
                        columns=['id','links']
                    )
                        .loc[:,['id','links']])
        else:
            raise ValueError


class Sentence:

    def __init__(self, owner, id, start, end, text):

        for field in ['owner', 'id','start','end','text']:
            exec(f'self.{field} = {field}')
        self.speaker = None

    def __repr__(self):
        return f'Sentence({self.id},({self.start},{self.end}), {self.text}, {self.speaker})'

    def __contains__(self, mention):
        "tell whether you cover the given mention"

        return self.start <= mention.start and self.end >= mention.end

    def set_speaker(self, speaker):
        self.speaker=speaker

class Mention:

    def __init__(self, args_dict):
        if self.valid_args(args_dict):
            self.__dict__.update(args_dict)
            self.sentence = self.get_sentence()
            self.sentence_id = self.sentence.id
            self.speaker = self.sentence.speaker
        else:
            logger.error(f"Mention init with {args_dict}")

    def __repr__(self):
        return f"Mention({self.id},{self.start},{self.end},{self.text})"

    def valid_args(self,args):
        return\
                isinstance(args,dict)\
                and set(args.keys()) == set(['id','tag','text','owner','start','end','role','form','coref'])

    def get_sentence(self):
        try:
            return\
                    [sentence
                     for sentence in self.owner.sentences
                     if self in sentence][0]
        except IndexError:
            logger.error(f"Could not get sentence for {self} in {self.owner}.")
            return Sentence(self, -1,-1,-1,'ERROR')
