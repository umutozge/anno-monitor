from labelbox import Client
from labelbox.exceptions import ResourceNotFoundError
import pandas as pd
import datetime

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

    def load_data(self, filename):

        client = Client(self.api_key)

        new_dataset = client.create_dataset(name=filename)


        # Create data payload
        # Use global key, a unique ID to identify an asset throughout Labelbox workflow. Learn more: https://docs.labelbox.com/docs/global-keys
        # You can add metadata fields to your data rows. Learn more: https://docs.labelbox.com/docs/import-metadata
        row_data = ""
        with open(filename, 'r', encoding="utf-8") as inf:
            row_data = json.load(inf)[0]['row_data']



        assets = [
            {
                "row_data": row_data,
                "global_key": filename[:-5],
                "media_type": "TEXT",
            }
        ]

        # Bulk add data rows to the dataset
        task = new_dataset.create_data_rows(assets)


        task.wait_till_done()
        print(task.errors)


if __name__=='__main__':
    from commons import LB_API_KEY, DATAPATH
    lbw = LBWorker(LB_API_KEY, DATAPATH)
