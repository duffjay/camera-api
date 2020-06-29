import sys

import hue_util
import settings


def main():
    # args
    config_filename = sys.argv[1]   # 0 based
    settings.init(config_filename)

    # get the bridge
    hue = hue_util.get_bridge(settings.hue_bridge_ip, settings.hue_bridge_username)
    print ("logged into the HUE bridge successfully")

    # get the collection of lights
    print ("\nhue - lights:")
    lights = hue_util.get_lights(hue)
    lights_bar = hue_util.get_light_ids(lights, "bar")
    print (f'bar light ids: {lights_bar}')

    # get the collection of groups
    print ("\nhue - groups")
    groups = hue_util.get_groups(hue)
    group_bar_id = hue_util.get_group_id(groups, "Bar")
    print (f'Bar group id: {group_bar_id}')

    # turn a group on
    group_bar = groups[group_bar_id]
    print (group_bar.data['action']['on'])
    response = hue_util.turn_group_on(group_bar, bri=254)

    # turn on a group
    
    #group_bar = hue_util.turn_on_group(group_bar_id, bri)
    




if __name__ == "__main__":
    main()