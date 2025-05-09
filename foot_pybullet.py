'''
角度の定義方向が終わっているので要調整．
'''

from scipy.optimize import newton
import pybullet as p
import time, json
import pybullet_data as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from foot_trajectory.foot_trajectory import (
    SplineFootTrajectoryGenerator, 
    SineFootTrajectoryGenerator, 
    BezierFootTrajectoryGenerator,
)

f = open("parameters.json", 'r')
json_param = json.load(f)["simQ"]
parameters = {"L1": json_param["L1"],
            "L2": json_param["L2"],
            "L3": json_param["L3"],
            "theta1_limit": json_param["theta1_limit"],
            "theta2_limit": json_param["theta2_limit"],
            "theta3_limit": json_param["theta3_limit"],
            "leg_deviation":json_param["leg_deviation"],
            "height": json_param["height"]
}

num_points = 50  # 計算する軌道上の点の数
rad_offset = 0
# orbit_tag = "ellipse"
orbit_tag = "triangle"
orbit_tag = "sin"
# orbit_tag = "spline"
orbit_tag = "bezier"

textureId = -1
useProgrammatic = 0
useTerrainFromPNG = 1
useDeepLocoCSV = 2
updateHeightfield = False
# heightfieldSource = useDeepLocoCSV
heightfieldSource = useTerrainFromPNG
random.seed(10)
heightPerturbationRange = 0.05

# 関節角度（ラジアン）
theta1 = np.radians(0)
theta2 = np.radians(0)
theta3 = np.radians(0)
target_position_ox = 0
target_position_oy = 0
target_position_oz = 0
x0=x1=x2=x3=y0=y1=y2=y3=z0=z1=z2=z3 = 0
theta_dict = {    
    "theta1": theta1,
    "theta2": theta2,
    "theta3": theta3
    }
# 3Dロボットのリンク長
L1, L2, L3 = parameters["L1"], parameters["L2"], parameters["L3"]
reset_num = 1 # pybulletの描画をリセットするための変数

# シミュレーションを描画する順運動学
def forward_kin(theta1, theta2, theta3):
    global x0,x1,x2,x3,y0,y1,y2,y3,z0,z1,z2,z3
    x1 = 0
    y1 = L1 * np.cos(theta1)
    z1 = - L1 * np.sin(theta1)

    x2 = L2 * np.sin(theta2)
    y2 = y1 + L2 * np.cos(theta2) * np.sin(theta1)
    z2 = z1 + L2 * np.cos(theta2) * np.cos(theta1)

    x3 = x2 + L3 * np.sin(theta2 + theta3)
    y3 = y2 + L3 * np.cos(theta2 + theta3) * np.sin(theta1)
    z3 = z2 + L3 * np.cos(theta2 + theta3) * np.cos(theta1)
    print("グラフ：",x3,",",y3,",",z3)
def inverse_kinematics(parameters, target_position):
    global theta1, theta2, theta3
    L1, L2, L3 = parameters["L1"], parameters["L2"], parameters["L3"]
    X, Y, Z = target_position[0], target_position[1], target_position[2]
    R = (Y**2+Z**2)
    h = np.sqrt(Y**2+Z**2-L1**2)
    Rp = np.sqrt(X**2+h**2)
    # theta1 = np.arccos((L1*Y+h*Z)/R) # 合ってるけどarccosやから定義域が面倒なので今回はパス
    theta1 =np.arctan2(h, L1) - np.arctan2(Z, Y) # これは合ってる
    # theta1 = np.arcsin((h*Y-L1*Z)/R)             # これもなぜか合ってる
    theta2 = np.arctan2(X, h) - np.arccos((-L3**2+Rp**2+L2**2)/(2*L2*Rp)) # θ3が０以下にならない前提（arccosの性質）
    theta3 = np.radians(180) - np.arccos((L2-h*np.cos(theta2)-X*np.sin(theta2))/L3) # なんで180から引かないといけないのかは不明

    print("check:", np.degrees(theta1), np.degrees(theta2), np.degrees(theta3))
    thetas = (theta1, theta2, theta3)
    for i in range(3):
        key = f"theta{i+1}"
        theta_dict[key] = thetas[i]
# ケプラー方程式を解いて離心近点角 (Eccentric Anomaly) を求める
def __kepler(E, M, e):
    return E - e * np.sin(E) - M  # ケプラーの方程式
def calc_orbit(num_points, rad_offset):
    global target_position_ox, target_position_oy, target_position_oz
    target_position_oy = L1
    M = np.linspace(0-rad_offset, 2 * np.pi-rad_offset, num_points*2)

    # 楕円
    if orbit_tag == "ellipse":
        a = 0.8 # 半長軸
        e = 0.8  # 離心率
        # 平均近点角 (Mean Anomaly) を 0 から 2π まで分割(for i in range(num_points):i*(2π-0)/num_points  )
        # 離心近点角 E をニュートン法で解く
        E = np.zeros(num_points)  # 配列を初期化
        for i in range(num_points):
            E[i] = newton(__kepler, M[i], args=(M[i], e))
        # 楕円軌道の極座標
        r = a * (1 - e * np.cos(E))  # 軌道半径
        theta = np.arctan2(np.sqrt(1 - e**2) * np.sin(E), np.cos(E) - e)  # 真近点角
        # 楕円軌道の直交座標
        target_position_ox = r * np.cos(theta) + a
        target_position_oz = r * np.sin(theta) + (L2+L3-a)
        print(type(target_position_ox))

    # 三角波
    elif orbit_tag == "triangle":
        # 三角形の頂点 (例: 正三角形)
        A = np.array([1, L2+3*L3/4])
        B = np.array([-1, L2+3*L3/4])
        C = np.array([0, L2+L3/2])

        # 三角形の辺 (AB → BC → CA の順で移動)
        edges = [ (A, B), (B, C), (C, A) ]

        steps_per_edge = int(num_points/3) + 1  # 各辺の分割数

        # 軌道生成
        target_position_ox = []
        target_position_oz = []

        # 軌道生成
        for start, end in edges:
            for t in np.linspace(0, 1, steps_per_edge, endpoint=False):
                point = (1 - t) * start + t * end  # 線形補間
                target_position_ox.append(point[0])
                target_position_oz.append(point[1])        

    # sin軌道
    elif orbit_tag == "sin":
        sine_generator = SineFootTrajectoryGenerator(base_frequency=0.2, initial_phi=0.0)
        sine_trajectory = sine_generator.compute_trajectory(M, frequency_offset=0, width=3, height=1.5)
        target_position_ox = sine_trajectory[0]
        target_position_oz = - sine_trajectory[1] + L2+3*L3/4
        print("sin")

    # スプライン軌道
    elif orbit_tag == "spline":
        spline_generator = SplineFootTrajectoryGenerator(base_frequency=0.2, initial_phi=0.0)
        spline_trajectory = spline_generator.compute_trajectory(M, frequency_offset=0, width=3, height=1.5)
        target_position_ox = spline_trajectory[0]
        target_position_oz = - spline_trajectory[1] + L2+3*L3/4

    # ベジェ軌道
    elif orbit_tag == "bezier":
        bezier_generator = BezierFootTrajectoryGenerator(base_frequency=0.2, initial_phi=0.0)
        bezier_trajectory = bezier_generator.compute_trajectory(M, frequency_offset=0, width=2, height=1.5)  
        target_position_ox = bezier_trajectory[0]
        target_position_oz = - bezier_trajectory[1] + L2+3*L3/4
def wrap_list(now_phase, num_points):
    if now_phase > num_points - 1:
        # print(now_phase - (num_points - 1))
        return now_phase - (num_points - 1)
    else:
        return now_phase
def input_foot_orbit(num_points, parameters):
    for i in range(num_points):
        print(orbit_tag)
        delay_phase = wrap_list(i + int(num_points/2), num_points)
        target_position_rf = (target_position_ox[delay_phase], target_position_oy, target_position_oz[delay_phase])
        target_position_lr = (target_position_ox[delay_phase], target_position_oy, target_position_oz[delay_phase])
        target_position_rr = (target_position_ox[i], target_position_oy, target_position_oz[i])
        target_position_lf = (target_position_ox[i], target_position_oy, target_position_oz[i])
        target_positions = [target_position_rf, target_position_rr, target_position_lf, target_position_lr]

        # 足先軌道を代入し運動学を計算
        for k in range(4):
            inverse_kinematics(parameters, target_positions[k])
            for j in range(3):
                key = f"theta{j+1}"
                target_pos = theta_dict[key]
                if k < 2:
                    target_pos = - theta_dict[key]
                # print(type(target_positions[k]))
                p.setJointMotorControl2(robotId, 4 * k + j,
                                            p.POSITION_CONTROL,
                                            targetPosition = target_pos)

        # 座標をプロット
        print("目標座標：",target_position[0], target_position[1], target_position[2])
        p.stepSimulation()
        time.sleep(0.03)

calc_orbit(num_points, 0)
forward_kin(theta1, theta2, theta3)
target_position = [x3, y3, z3]

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pd.getDataPath())
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

if heightfieldSource==useDeepLocoCSV:
  terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[.5,.5,1],fileName = "heightmaps/ground0.txt", heightfieldTextureScaling=128)
  terrain  = p.createMultiBody(0, terrainShape)
  p.resetBasePositionAndOrientation(terrain,[0,0,0], [0,0,0,1])
if heightfieldSource==useTerrainFromPNG:
  terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[.1,.1,0.6],fileName = "heightmaps/gimp_overlay_out.png", heightfieldTextureScaling=128)
  # terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[.1,.1,0.6],fileName = "heightmaps/wm_height_out.png", heightfieldTextureScaling=128)
  textureId = p.loadTexture("heightmaps/gimp_overlay_out.png")
  terrain  = p.createMultiBody(0, terrainShape)
  p.changeVisualShape(terrain, -1, textureUniqueId = textureId)
p.changeVisualShape(terrain, -1, rgbaColor=[1,1,1,1])

p.setAdditionalSearchPath(pd.getDataPath()) #optionally
p.setGravity(0,0,-10)
# planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
# testPos = [-10,0,1]
testPos = [-1,-1,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
# PyBulletのGUIを起動してアクティブにしておく
robotId = p.loadURDF("urdf/model.urdf",startPos, startOrientation)
# objId = p.loadURDF("samurai.urdf",testPos, startOrientation)
# objId = p.loadURDF("sphere2.urdf",testPos, startOrientation)
# ロボットの位置/姿勢を設定可能なデバッグ用のスライダーを定義
reset_button = p.addUserDebugParameter("reset_button", 1, 0, 0)
Id_0 = p.addUserDebugParameter("Num0", -1.57, 1.57, 0)
Id_1 = p.addUserDebugParameter("Num1", -1.57, 1.57, 0)
Id_2 = p.addUserDebugParameter("Num2", -1.57, 1.57, 0)
Id_3 = p.addUserDebugParameter("Num3", -1.57, 1.57, 0)
Id_4 = p.addUserDebugParameter("Num4", -1.57, 1.57, 0)
Id_5 = p.addUserDebugParameter("Num5", -1.57, 1.57, 0)
Id_6 = p.addUserDebugParameter("Num6", -1.57, 1.57, 0)
Id_7 = p.addUserDebugParameter("Num7", -1.57, 1.57, 0)
Id_8 = p.addUserDebugParameter("Num8", -1.57, 1.57, 0) # 向き逆， 右前足のθ1
Id_9 = p.addUserDebugParameter("Num9", -1.57, 1.57, 0)
Id_10 = p.addUserDebugParameter("Num10", -1.57, 1.57, 0)
Id_11 = p.addUserDebugParameter("Num11", -1.57, 1.57, 0)
Id_12 = p.addUserDebugParameter("Num12", -1.57, 1.57, 0)
Id_13 = p.addUserDebugParameter("Num13", -1.57, 1.57, 0)
Id_14 = p.addUserDebugParameter("Num14", -1.57, 1.57, 0)
IDs = [Id_0, Id_1, Id_2, Id_3, Id_4, Id_5, Id_6, Id_7, Id_8, Id_9, Id_10, Id_11, Id_12, Id_13, Id_14]
pre_link_data=(0,0,0)

while True:
    p.stepSimulation()
    # print(p.readUserDebugParameter(reset_button))
    # ロボットモデルのみリセットする
    if(p.readUserDebugParameter(reset_button) == reset_num):
        # 既存のオブジェクトを削除しリロード
        p.removeBody(robotId)
        robotId = p.loadURDF("urdf/model.urdf",startPos, startOrientation)
        reset_num = reset_num + 1
        time.sleep(2)
    time.sleep(0.03)
    input_foot_orbit(num_points, parameters)
