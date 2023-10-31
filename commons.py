from random import randint

USERS = [{"name":"faruk", "id":1, "email":"buyuktekinf@gmail.com"},
         {"name":"umut",  "id":2, "email":"umozge@metu.edu.tr"},
         {"name":"anil",  "id":3, "email":"aogdul@gmail.com"},
         {"name":"derin", "id":4, "email":"derinxdincer@gmail.com"},
         {"name":"batuhan","id":5, "email":"batuhan.karatas@metu.edu.tr", "email2":"batuhan.karatas94@gmail.com"}]


def focus_on(table: list[dict], field_name: str) -> dict:
    """Transforms a list like:
        [{a:x_1, b:y_1, c:z_1,...},
         {a:x_2, b:y_2, c:z_2,...},
         ...]

         to

         {y_1:x_1, y_1:x_1, z_1:x_1,...,y_2:x_2, z_2:x_2,... }

         where field_name=a, and x's and y's need not be unique.

         This forms a mapping from a table where all the values in a row are mapped to the value of a certain field in that row.
        """

    from functools import reduce

    return reduce(lambda x,y:x|y,map(lambda entry : {val:entry[field_name] for key, val in entry.items() if key != field_name},
                                     table))


def make_color_picker(colors=['green','blue','orange','red']):
    last_pick = colors[randint(0, len(colors) -1)]

    def func():
        nonlocal last_pick
        pick = colors[randint(0, len(colors) -1)]
        if pick != last_pick:
            last_pick = pick
            return pick
        else:
            return func()

    return func


USERS = focus_on(USERS,'name')

DATAPATH = 'data'

LB_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGdkZThvaGEwYmcxMDcxeGI3eXkyaGxyIiwib3JnYW5pemF0aW9uSWQiOiJjbGdkZThvZ3gwYmcwMDcxeGYydGwzeDl1IiwiYXBpS2V5SWQiOiJjbGdldGQyeW8xM3I0MDd3ZDhkbWc3ZGpmIiwic2VjcmV0IjoiMjc4YTAwZmQwNDkzOGRjMjRkYzdkMDVjODY3NjcxZDgiLCJpYXQiOjE2ODEzNzE4MDksImV4cCI6MjMxMjUyMzgwOX0.D4McllRdFhPUDo7WJBZJLEMPVTX8uX3mEeACWbozmas'
LB_PROJECTS = ['clo7mc56c057507yzbjsgf979','clo7mckef055c071oc9qd98qt','clo7metbw0blp07yp004dfg50']


LSKEY = '0272f4ea5b7a6f6d1567943b9c3ec79a91b66cd0'
LS_AUTH_HEADER = {'Authorization': f'Token {LSKEY}'}
LS_SERVER = 'http://lfcs.ii.metu.edu.tr:8080'
