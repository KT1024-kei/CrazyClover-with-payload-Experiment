#!/user/bin/env python
# coding: UTF-8

import yaml

# ワールド座標系と剛体の座標系のフレーム名を設定
class Frames_setup(object):

    def __init__(self):

        # ワールド座標 の定義
        self.world_frame = "world"
        # エージェント座標の定義
        self.children_frame = []

        # crazyflies.yamlに登録されているcrazyflieのidを取得
        with open("/home/user/crazyswarm/ros_ws/src/crazyswarm/launch/crazyflies.yaml", 'r') as ymlfile:
            root = yaml.load(ymlfile)
            for node in root["crazyflies"]:
                # すべての子フレームを登録
                self.children_frame.append("cf" + str(node["id"]))

        print("World frame name :{}".format(self.world_frame))
        print("children frame names :{}".format(self.children_frame))

'''
if __name__ == '__main__':
    Frames_setup()
'''
