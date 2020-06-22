from hueber.api import Bridge
from hueber.lib import LightBuilder

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

def get_lights_by_id(hue):
    return

def get_lights_by_name(hue):
    return

def get_groups(hue):
    return

def get_light_data(light_id):
    return