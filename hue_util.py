import os
import sys

# add the (private) Hue project to the Python path
cwd = os.getcwd()
hue_path = os.path.abspath(os.path.join(cwd, '..', 'hueber'))
sys.path.append(hue_path)

# clone the project:  https://github.com/mbaltrusitis/hueber
# pip install hueber
from hueber.api import Bridge
from hueber.lib import LightBuilder
from hueber.lib import GroupBuilder

# 3 pre-requisites
#
# - Make sure your Hue bridge is working
# - set a static IP address
#   https://notsealed.com/how-to-set-static-philips-hue-bridge-ip-address-fix.html
# - create a username
#   https://developers.meethue.com/develop/get-started-2/
#
# 

def get_bridge(ip, username):
    '''
    you need a stable IP and the user name
    - create the username from the app, it's really long
    - this is the security
    '''
    hue = Bridge(ip, username)
    return hue

def get_lights(hue):
    return hue.lights

def get_light_ids(lights, name):
    matched_light_ids = []
    for light_id in lights:
        if lights[light_id]['name'].startswith(name):
            matched_light_ids.append(light_id)
            match = '***'
        else:
            match = '   '

        print (f'{match} Light# {light_id} - name: {lights[light_id]["name"]} type: {lights[light_id]["type"]}')
    return matched_light_ids


def get_lights_by_name(hue):
    return

def get_groups(hue):
    return hue.groups

def get_group_id(groups, name):
    matched_group_id = None
    for group_id in groups:
        if name == groups[group_id]['name']:
            matched_group_id = group_id
            match = '***'
        else:
            match = '   '

        print (f'{match} Group# {group_id} - name: {groups[group_id]["name"]} type: {groups[group_id]["type"]}')
    return matched_group_id

def turn_group_on(group, bri=254):
    new_update = GroupBuilder()
    new_update["on"] = True
    new_update["bri"] = bri
    response = group.push(new_update.update_str())
    return response

def turn_group_off(hue, group_id):
    new_update = GroupBuilder()
    new_update["on"] = True
    new_update["bri"] = bri
    response = hue.groups[group_id].push(new_update.update_str())
    return response

def get_light_data(light_id):
    return