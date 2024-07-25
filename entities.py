import pandas as pd
import numpy as np
import math
import os
import sys
import re
import json
import time
import copy

import random

from functools import reduce

from commons import USERS, LB_API_KEY, DATAPATH, LB_PROJECTS, make_color_picker, convert_link_tag, make_counter, shrink_space
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

COLOR = make_color_picker()

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
                if dialog.is_ready:
                    store.append(dialog)

        self.dialogs = (pd.DataFrame({'object': store})
                        .assign(name = lambda df: df['object'].apply(lambda row: row.name))
                        .assign(datarow_id = lambda df: df['object'].apply(lambda row: row.datarow_id))
                        .assign(project_id = lambda df: df['object'].apply(lambda row: row.project_id))
                        )
        return self

    def summarize(self):
        table=\
                (pd.DataFrame([dialog.__dict__ for dialog in self.dialogs['object'].values])
                 .assign(**{'#words' : lambda df:
                         df.apply(lambda row: row['owner'].get_dialog('name',row['name']).get_word_count(),
                                 axis=1)})
                 .assign(**{'#sentences' : lambda df:
                         df.apply(lambda row: row['entity_grid'].get_sentence_count(),
                                 axis=1)})
                 .assign(**{'#mentions' : lambda df:
                         df.apply(lambda row: row['entity_grid'].get_mention_count(),
                                 axis=1)})
                 .assign(**{'#entities' : lambda df:
                         df.apply(lambda row: len(row['entity_grid'].get_chains()),
                                 axis=1)})
                 .assign(**{'mention/entity': lambda df:
                         df.apply(lambda row: row['#mentions']/row['#entities'],
                                 axis=1)})
                 .assign(**{'%null': lambda df:
                          df.apply(lambda row:
                                   (row['entity_grid']['mentions']['form'].value_counts(normalize=True) *100)['null'], axis=1)})
                 .assign(**{'%overt': lambda df:
                          df.apply(lambda row:
                                   (row['entity_grid']['mentions']['form'].value_counts(normalize=True) *100)['overt'], axis=1)})
                 .assign(**{'%subject': lambda df:
                          df.apply(lambda row:
                                   (row['entity_grid']['mentions']['role'].value_counts(normalize=True) *100)['subj'], axis=1)})
                 .assign(**{'%non-subject': lambda df:
                          df.apply(lambda row:
                                   (100 - row['%subject']), axis=1)})
                 .assign(**{'%null-in-subj': lambda df:
                            df.apply(lambda row:
                                     (row['entity_grid']['mentions'].loc[lambda x: x['role'] == 'subj']['form'].value_counts(normalize=True) *100)['null'],
                                     axis=1)})
#                 .pipe(lambda df: print(df.apply(lambda row:
#                                     [row['entity_grid']['mentions'].loc[lambda x: x['role'] != 'subj']['form'].unique(),row['entity_grid'].dialog],
#                                     axis=1)) or df)
                 .assign(**{'%null-in-non-subj': lambda df:
                            df.apply(lambda row:
                                     (row['entity_grid']['mentions'].loc[lambda x: x['role'] != 'subj']['form'].value_counts(normalize=True) *100)['null'],
                                     axis=1)})
                 .loc[:,['name','labelers','#words','#sentences','#mentions','#entities','mention/entity','%null','%overt','%subject','%non-subject','%null-in-subj','%null-in-non-subj']]
                )
        
        return table, table.describe()

    def update(self, dialog):
        self.data_reader.read_data(dialog.project_id, update=True)
        self.load_projects()

    def update_all(self):
        self.load_projects(update=True)

    def get_dialog(self, key, value):
        return self.dialogs.loc[lambda df: df[key] == value, 'object'].values[0]

    def __len__(self):
        return (len(self.dialogs))

    def __bool__(self):
        return bool(len(self))

    def write_grids(self):
        self.load_projects()
        with open('grids.json', 'w', encoding="utf-8") as ouf:
            ouf.write(json.dumps([dialog.entity_grid.to_dict() for dialog in self.dialogs.loc[:, "object"]], indent=4, ensure_ascii=False))

class Dialog:

    def __init__(self, lb_datarow, owner):

        self.owner = owner
        self.lb_datarow = lb_datarow
        self.datarow_id = lb_datarow['data_row']['id']
        self.text = lb_datarow['data_row']['row_data']
        self.name = lb_datarow['data_row']['details']['dataset_name'][:-5]
        self.project_id = list(lb_datarow['projects'].keys())[0]
        self.is_ready = False

        self.spans, self.relations = self.organize_data()


        if self:
            self.labelers = list(self.spans.loc[:,'labeler'].unique())
            self.agreement = Agreement(self)
            self.indexed_text = self.generate_indexed_text()
            self.write_to_disk()
            self.entity_grid=EntityGrid(self)
            logger.info(f'Formed {self}')
            self.is_ready = True

    def __repr__(self):
        return f"Dialog({self.name}, {self.project_id}, {self.datarow_id})"

    def __bool__(self):
        return\
                self.spans is not None and\
                self.relations is not None and\
                len(self.spans.loc[:,'labeler'].unique()) == 2 and\
                len(self.relations.loc[:,'labeler'].unique()) == 2


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
            logging.warning(f"Not enough labeler in {self.name}")
            return None, None

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

        spans = (raw_spans
                 .assign(id = lambda df: df.loc[:,'range'].apply(lambda x: range_to_id[x]))
                 .assign(text = lambda df: df['range'].apply(lambda x: self.text[x[0]:x[1]]))
                 .drop('range', axis=1)
                 .reset_index(drop=True))

        relations =\
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

        return spans, relations


    def generate_indexed_text(self):

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
                accu += f':{COLOR()}['
                last_edit_point = start_or_end[0]


        if last_edit_point < len(text):
            accu += text[last_edit_point:]

        accu = accu.replace('p:','p: ').replace('o:','o: ').replace('\n','\n<br/>')
        return accu

    def get_word_count(self):
        return len(self.text.split())

class Agreement:

    def __init__(self, dialog):

        self.dialog = dialog
        self.labelers = dialog.labelers
        self.spans = [dialog.spans[dialog.spans['labeler'] == labeler]
                      for labeler in self.labelers]

        self.relations = [dialog.relations[dialog.relations['labeler'] == labeler]
                      for labeler in self.labelers]

        self.results = {
            'matched': {
                'spans': self._compute_matched_spans(),
                'relations':self._compute_matched_relations()},
            'unmatched': {
                'spans': self._compute_unmatched_spans(),
                'relations': self._compute_unmatched_relations()},
            'disagreed': {
                'spans': self._compute_disagreed_spans(),
                'relations': self._compute_disagreed_relations()} }

        self.summary = self._compute_summary()
        self.write_to_disk()

    def write_to_disk(self):
        result = {'spans': [json.loads(span.to_json(orient='records')) for span in self.spans],
                  'rels': [json.loads(relation.to_json(orient='records')) for relation in self.relations],
                  'mspans': json.loads(self.results['matched']['spans'].to_json(orient='records')),
                  'uspans': json.loads(self.results['unmatched']['spans'].to_json(orient='records')),
                  'mrels':  json.loads(self.results['matched']['relations'].to_json(orient='records')),
                  'urels':  json.loads(self.results['unmatched']['relations'].to_json(orient='records')),
                  'dspans':  json.loads(self.results['disagreed']['spans'].to_json(orient='records')),
                  'drels':  json.loads(self.results['disagreed']['relations'].to_json(orient='records')) }
        with open(f'agrdata/{self.dialog.name}.json', 'w', encoding='utf-8') as ouf:
            ouf.write(json.dumps(result, indent=4, ensure_ascii=False))

    def __bool__(self):
        return bool(self.results)

    def _compute_summary(self):

        return (pd.DataFrame(
            {'spans':
             {'matched': len(self.results['matched']['spans']),
              'unmatched': len(self.results['unmatched']['spans']),
              'disagreed': len(self.results['disagreed']['spans'])},
             'relations':
             {'matched': len(self.results['matched']['relations']),
              'unmatched':len(self.results['unmatched']['relations']),
              'disagreed':len(self.results['disagreed']['relations'])}})
            .pipe(lambda df: pd.concat([df, pd.DataFrame([df.sum()], index=['total'])]))
            .assign(spans_p = lambda df: df.apply(lambda row: f"{round(row['spans']/df.at['total','spans'] * 100)}%", axis=1),
             relations_p = lambda df: df.apply(lambda row: f"{round(row['relations']/df.at['total','relations'] * 100)}%", axis=1))
            .pipe(lambda df: df.iloc[:,[0,2,1,3]])
        )

    def _compute_matched_relations(self):

        return (pd.merge(*self.relations,
                         on=['from','to','tag'])
                .loc[:,['from','to','tag']]
                .reset_index(drop=True)
               )

    def _compute_unmatched_relations(self):

        return (pd.merge(*self.relations,
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

        return (pd.merge(*self.relations,
                         on=['from','to'],
                         how='inner',
                         suffixes=[f'_{l}' for l in self.labelers])
                .query(f'tag_{self.labelers[0]} != tag_{self.labelers[1]}')
                .sort_values(by="from")
                .pipe(lambda df: df.drop(columns=[column for column in df.columns if column.startswith('labeler')]))
                .reset_index(drop=True)
               )

    def _compute_matched_spans(self):

        return (pd.merge(*self.spans, on=['id','tag'], how='inner')
                .sort_values(by="id")
                .rename(columns={'start_x':'start','end_x':'end','text_x':'text'})
                .pipe(lambda df:
                      df.drop(columns=[column for column in df.columns if column[-2]=='_']))
                .reset_index(drop=True)
               )

    def _compute_unmatched_spans(self):

        return (pd.merge(*self.spans,
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

        return (pd.merge(*self.spans,
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
        self.spans, self.relations = self.create_gold(mode='length')
        self.text = shrink_space(self.dialog.text)
        self.sentences = self.acquire_sentences()
        self.links = self.acquire_links()
        self.mentions = self.acquire_mentions()

        # enrich_mentions depends on the existence of self.mentions
        self.mentions = {m.id:m for m in self.enrich_mentions()}
        self.coref_classes = self.create_coref_classes()
        self.chains = [Chain(x) for x in self.coref_classes]

    def create_coref_classes(self):

        mention_ids = {int(i):True for i in self.mentions.keys()}

        ccs = []
        trial_count = 100
        while mention_ids and trial_count > 0:
            for mid in mention_ids.keys():
                mention = self.get_mention(mid)
                if mention.is_head():
                    ccs.append(CorefClass(mention))
                    mention_ids[mid] = False
                else:
                    if any(map(lambda cc: cc.push(mention), ccs)):
                        mention_ids[mid] = False

            mention_ids = dict(filter(lambda x: x[1], mention_ids.items()))
            trial_count -= 1

        if mention_ids:
            logger.warn(f'Could not coref classify {list(mention_ids.keys())} in {self.dialog}')

        return ccs

    def get_mention(self, id):
        return self.mentions[id]

    def acquire_sentences(self):
        sentences=\
                [Sentence(self, *sent)
                 for sent in
                 list(
                     map(lambda x: (x[0],x[1][0],x[1][1],x[1][2]),
                         enumerate(
                             reduce(lambda result, sent:
                                    result + [(result[-1][1],
                                               (lambda x:
                                                (logger.error(f'Text search error in {self.dialog.name} result {result} and sentence {sent}.') or x)
                                                if x == -1 else x) (self.text.find(sent, result[-1][1]))
                                               +len(sent),
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
        """Returns a dict of the form:
            {from_id: [(to_id, tag),...]}
        """
        return\
                reduce(lambda dic, tup:
                       dic|{tup[0]:[tup[1]]} if tup[0] not in dic.keys()
                       else dic|{tup[0]: dic[tup[0]] + [tup[1]]},
                       [(rel['from'], (rel['to'],convert_link_tag(rel['tag'])))
                        for rel in self.relations.to_dict(orient='index').values()],
                       dict())

    def get_chains(self):
        return self.chains

    def get_mentions(self):
        return self.mentions.values()

    def get_mention_count(self):
        return len(self.get_mentions())

    def get_sentence_count(self):
        return len(self.sentences)

    def acquire_mentions(self):
        return\
                {mention.id:mention
                 for mention in
                 [Mention(record)
                  for record in
                  self.expand_mentions(
                      self.spans_to_mentions()
                  ).to_dict('records')]
                }

    def enrich_mentions(self):

        return\
                [mention.set_role().set_form().set_in_link()
                 for mention in
                 self.get_mentions() ]

    def spans_to_mentions(self):
        return\
                (pd.DataFrame(
                    [{'owner':self,
                      'coref': self.links.get(v['id']), }
                         |v
                     for  v in self.spans.to_dict(orient='index').values() ]
                )
                    .sort_values(by='start')
                )

    def expand_mentions(self, mentions):

        gen_id = make_counter(self.spans.loc[:,'id'].max())

        return\
                (pd.concat([mentions,
                           (mentions
                            .dropna(subset=['coref'])
                            .assign(coref = lambda df: df.apply(
                                lambda row: [row['coref'][1]]
                                if len(row['coref']) == 2 else None, axis=1))
                            .dropna(subset=['coref'])
                            .assign(id = lambda df: df.apply(
                                lambda row: gen_id(), axis=1))
                           )]
                         )
               #  .pipe(lambda df: print(df.to_string()) or df)
                 .assign(out_link = lambda df:
                         df.apply(lambda row:
                                  convert_link_tag(row['coref'][0][1])
                                  if row['coref'] else None, axis=1))
                 .assign(coref = lambda df:
                         df.apply(lambda row:
                                  row['coref'][0][0]
                                  if row['coref'] else None, axis=1))
                 .rename(columns={'coref':'ant'})
                )

    def create_gold(self, mode='random'):
        if mode=='strict':
            spans = self.dialog.agreement.results['matched']['spans']
            relations = self.dialog.agreement.results['matched']['relations']
            return spans, relations
        elif mode=='length':
            winner=max(self.dialog.labelers, key=lambda x: len(self.dialog.spans[self.dialog.spans.labeler==x]))
            spans = self.dialog.spans[self.dialog.spans.labeler==winner] 
            relations = self.dialog.relations[self.dialog.relations.labeler==winner]
            logger.info(f'Picked {winner} by length mode in dialog {self.dialog.name}.')
            return spans.drop(['labeler'],axis=1), relations.drop(['labeler'],axis=1)
        elif mode=='random':
            labeler = self.dialog.labelers[round(random.random())]
            spans = self.dialog.spans[self.dialog.spans.labeler == labeler]
            relations = self.dialog.relations[self.dialog.relations.labeler==labeler]
            logger.info(f'Picked {labeler} at random mode in dialog {self.dialog.name}.')
            return spans.drop(['labeler'],axis=1), relations.drop(['labeler'],axis=1)

    def to_dict(self):

        data = {}
        data["dialog"] = self.dialog.name
        data["mentions"] = [x.to_dict() for x in self.mentions.values()]
        data["sentences"] = [x.to_dict() for x in self.sentences]
        data["chains"] = [x.to_dict() for x in self.chains]
        data["text"] = self.text
        return data

    def __repr__(self):
        return\
                f"EntityGrid({self.dialog.name})"

    def __getitem__(self, field):
        """For viewing purposes only"""

        match field:
            case 'sentences':
                return pd.DataFrame(
                    [sent.__dict__
                     for sent in self.sentences],
                    columns=['id','start','end','text','speaker']
                )
            case 'mentions':
                return pd.DataFrame([mention.__dict__ for mention in self.get_mentions()],
                                    columns= ['id',
                                              'tag',
                                              'text',
                                              'sentence_id',
                                              'sentence_text',
                                              'speaker',
                                              'role',
                                              'form',
                                              'out_link',
                                              'in_link',
                                              'ant',
                                              'pre',
                                             ]).rename(columns={'sentence_id':'sent_id'})
            case 'links':
                return\
                        (pd.DataFrame(
                            [{'id': key, 'links':str(val)}
                             for key, val in self.links.items()],
                            columns=['id','links']
                        )
                            .loc[:,['id','links']])
            case 'coref_classes':
                return\
                        '\n\n'.join([x.__repr__() for x in self.coref_classes])
            case 'chains':
                return\
                        (pd.DataFrame([c.__dict__ for c in self.chains],
                                      columns=['root','sequence','length','roles','forms']
                                     ))
            case _:
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

    def to_dict(self):
        data = copy.copy(self.__dict__)
        data.pop('owner')
        return data

class Mention:

    names = ['id','tag','text','owner','start','end','role','form','ant','pre','out_link','in_link','sentence','sentence_id','sentence_text']

    def __init__(self, args_dict):

        self.__dict__ = {k:None for k in self.names}
        self.__dict__.update(args_dict)
        if len(self.__dict__.keys()) == len(self.names):
            self.__dict__.update(args_dict)
            self.sentence = self.get_sentence()
            self.sentence_id = self.sentence.id
            self.sentence_text = self.sentence.text
            self.speaker = self.sentence.speaker
        else:
            logger.error(f"Mention init with {args_dict}")

        self._fix_types()

    def _fix_types(self):

        self.id = int(self.id)

        for field in ['pre','ant']:
            if self.__dict__[field] is None or math.isnan(self.__dict__[field]):
                self.__dict__[field] = None
            else:
                self.__dict__[field] = int(self.__dict__[field])

    def __repr__(self):
        return f"Mention({self.id},{self.start},{self.end},{self.text},{self.owner.dialog.name})"

    def get_antecedent(self):
        try:
            return\
                    next(
                        filter(lambda mention: mention.id==self.ant,
                               self.owner.get_mentions()))
        except StopIteration:
            return None

    def sentence_mate(self, mention):
        return self.sentence_id == mention.sentence_id

    def precedes(self, mention):
        return self.start > mention.start

    def is_head(self):
        return not isinstance(self.ant, int)

    def get_sentence(self):
        try:
            return\
                    [sentence
                     for sentence in self.owner.sentences
                     if self in sentence][0]
        except IndexError:
            logger.error(f"Could not get sentence for {self} in {self.owner}.")
            return Sentence(self, -1,-1,-1,'ERROR')

    def get_grid_mates(self):
        return self.owner.get_mentions()

    def set_in_link(self):

        try:
            precedent =\
                    next(
                        filter(lambda mention: mention.ant == self.id,
                               self.get_grid_mates()))
        except StopIteration:
            self.in_link = None
            self.pre = None
        else:
            self.pre = precedent.id
            self.in_link = precedent.out_link

        return self

    def set_role(self):
        self.role = Linger.role(self)
        return self

    def set_form(self):
        self.form = Linger.form(self)
        return self

    def to_dict(self):
        data = copy.copy(self.__dict__)
        data.pop('owner')
        data.pop('sentence_text')
        data['sentence'] = data['sentence'].to_dict()
        return data

class Linger:

    @staticmethod
    def case(mention: pd.core.series.Series):
        if bool(
            re.search('n?(u|ü|i|ı)n', mention.owner.text[mention.end : mention.end+4])
        ):
            return 'gen'
        elif mention.owner.text[mention.end] == ' ':
            return 'nom'
        else:
            return 'obj'


    @staticmethod
    def is_null(mention: pd.core.series.Series):
        return\
                mention.text == ' ' or\
                mention.tag == 'pred'

    @staticmethod
    def form(mention: pd.core.series.Series):
        try:
            if mention.text == ' ' or mention.tag == 'pred':
                return 'null'
            elif mention.tag == 'nom' and\
                    mention.out_link == 'poss' and \
                    not mention.sentence_mate(mention.get_antecedent()):
                return 'null'
            else:
                return 'overt'
        except StopIteration:
            return 'iter-error'
        except AttributeError:
            return 'null'

    @staticmethod
    def role(mention: pd.core.series.Series):
        if mention.tag == 'nom':
            if Linger.case(mention) == 'gen':
                return 'poss' if mention.in_link == 'poss' else 'subj'
            elif Linger.case(mention) == 'nom':
                return 'subj'
            elif mention.text == ' ':
                return 'obj'
            elif Linger.case(mention) == 'obj':
                return 'obj'
        elif mention.tag == 'pred':
            if mention.out_link == 'subj':
                return 'subj'
            elif mention.out_link == 's_subj':
                return 's_subj'
            else:
                #print(mention.out_link)
                #print(mention)
                return 'unk'
        elif mention.tag == 'exo':
            match Linger.case(mention):
                case 'gen':
                    return 'poss'
                case 'nom':
                    return 'subj'
                case 'obj':
                    return 'obj'

class CorefClass:

    def __init__(self, mention):

        assert(mention.is_head())
        # data is a dict mapping mention id to mention
        self.mentions = {mention.id:mention}

    def hosts(self, mention):
        "tell whether the mention has an antecedent in this class"
        value = list(filter(lambda m: m.id == mention.ant,
                           self.mentions.values()))

        return bool(value)


    def push(self, mention):
        "push the mention if it fits in"
        if self.hosts(mention):
            self.mentions[mention.id] = mention
            return True
        else:
            return False

    def __repr__(self):
        return f"CorefClass({list(self.mentions.keys())})"

class Chain:


    def __init__(self, corefclass):

        self.corefclass = corefclass
        self.mentions = self.sort_mentions()
        self.root = self.get_root()
        self.mentions = list(filter(lambda x: not x.is_head(), self.mentions))
        self.sequence = [x.id for x in self.mentions]
        self.length = len(self.mentions)
#        assert(self.length == len(corefclass.mentions.values()))
        self.roles = [x.role for x in self.mentions]
        self.forms = [x.form for x in self.mentions]
        self.speakers = [x.speaker for x in self.mentions]

    def sort_mentions(self):
        return sorted(self.corefclass.mentions.values(), key=lambda x: x.id)

    def get_root(self):
        return next(filter(lambda x: x.is_head(), self.mentions))

    def to_dict(self):
        return\
        {"root":self.root.id,
         "corefs":  [x.id for x in
                     list(filter(lambda x: isinstance(x.ant, int), self.mentions))
                    ]}
