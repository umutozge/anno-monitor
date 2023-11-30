from labelbox import Client
from labelbox.exceptions import ResourceNotFoundError
import pandas as pd

import os
import sys
import json

import logging

logging.basicConfig(
    filename='logfile.log',
    encoding='utf-8',
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class LBWorker():

    def __init__(self, api_key, datapath):

        self.api_key=api_key
        self.datapath = datapath


    def _remote_read_data(self, project_id):
        logger.info(f"Reading {project_id} from remote.")
        client = Client(api_key=self.api_key)

        try:
            project = client.get_project(project_id)
        except ResourceNotFoundError as e:
            logger.error(e)
            return None

        export_params= {
            "attachments": False,
            "metadata_fields": False,
            "data_row_details": True,
            "project_details": False,
            "label_details": True,
            "performance_details": False
        }

        export_task = project.export_v2(params=export_params)

        export_task.wait_till_done()

        if export_task.errors:
            sys.stderr.write(f"\nErrors during export {export_task.errors}")

        export_json = export_task.result

        with open(os.path.join(self.datapath,f'.{project_id}.json'), 'w', encoding='utf-8') as ouf:
            json.dump(export_json,ouf,ensure_ascii=False, indent=4)


        logger.info(f"Read {project_id} from remote.")
        return export_json

    def _local_read_data(self, project_id):

        try:
            with open(os.path.join(self.datapath,f'.{project_id}.json'), 'r', encoding='utf-8') as inf:
                return json.load(inf)
        except json.decoder.JSONDecodeError as e:
            return self._remote_read_data(project_id)
        except FileNotFoundError:
            return self._remote_read_data(project_id)


    def read_data(self, project_id, update=False):

        if update:
            return self._remote_read_data(project_id)
        else:
            return self._local_read_data(project_id)
