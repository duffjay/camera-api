import hue_util
import settings


def main():
    # args
    config_filename = sys.argv[1]   # 0 based
    settings.init(config_filename)

    # get the bridge
    hue = hue_util.get_bridge(settings.hue_ip, settings.hue_username)

    # get the collection of lights

    # get the collection of groups


    # get light status from the camera is_color value
    # - if it is NOT color, it's dark and we want to turn on the light


if __name__ == "__main__":
    main()