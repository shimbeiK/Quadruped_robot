'''
座標を指定し，それに基づいて逆運動学を計算
計算結果を描画

今回の敗因は角度の定義域を考えずに計算していたこと
気づくタイミングはあったが，それ以外にも問題があったため解決に至らず
幾何学的に解いたことで値域が明確になり，解明に至る

ただ以下の点に注意すること
・±hをhでおいている
・θ3の定義域は0~180（後ろに回転しない）

・θ3が-75.000000000001みたいな値になってnanになるので丸める処理を挟む ： 確認できず
・θ2がnanになってるのでそこも修正 ： 端っこでしかnanを確認できず ⇒ 境界条件で済む？

楕円軌道だと遊脚相が曖昧になるため前に進まなくなる
三角形だと遊脚相がしっかりとれるため前に進むが，ペース歩容だと厳しい
'''
from scipy.optimize import newton
import pybullet as p
import time, json
import matplotlib.pyplot as plt
import numpy as np
import math as m
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

xmin,xmax=-2,6
ymin,ymax=-4,4
zmin,zmax=-1,7

rad_offset = 0
orbit_tag = "ellipse"
orbit_tag = "triangle"

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

# 関節角度（ラジアン）
theta1 = np.radians(0)
theta2 = np.radians(0)
theta3 = np.radians(0)
target_position_ox = 0
target_position_oy = 0
target_position_oz = 0
x0=x1=x2=x3=y0=y1=y2=y3=z0=z1=z2=z3 = 0

# 3Dロボットのリンク長
L1, L2, L3 = parameters["L1"], parameters["L2"], parameters["L3"]
num_points = 50  # 計算する軌道上の点の数

# 各関節位置（順運動学）
def init():
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.invert_yaxis()  
    ax.invert_zaxis()
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title("3DOF Robot Arm")
    ax.legend()

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
    # theta2, theta3 = -theta2, -theta3
    print("check:", np.degrees(theta1), np.degrees(theta2), np.degrees(theta3))

# ケプラー方程式を解いて離心近点角 (Eccentric Anomaly) を求める
def __kepler(E, M, e):
    return E - e * np.sin(E) - M  # ケプラーの方程式

def calc_orbit(num_points, rad_offset):
    global target_position_ox, target_position_oy, target_position_oz
    target_position_oy = L1

    # 楕円
    if orbit_tag == "ellipse":
        a = 0.8 # 半長軸
        e = 0.8  # 離心率
        # 平均近点角 (Mean Anomaly) を 0 から 2π まで分割(for i in range(num_points):i*(2π-0)/num_points  )
        M = np.linspace(0-rad_offset, 2 * np.pi-rad_offset, num_points)
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
        A = np.array([1, L2+L3/2])
        B = np.array([-1, L2+L3/2])
        C = np.array([0, L2])

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

    
    # スプライン軌道
    
    # ベジェ軌道

calc_orbit(num_points, np.pi/2)

def draw_foot_orbit(num_points, parameters):
    for i in range(num_points):
        target_position = (target_position_ox[i], target_position_oy, target_position_oz[i])

        # 足先軌道を代入し運動学を計算
        inverse_kinematics(parameters, target_position)
        forward_kin(theta1, theta2, theta3)

        # 座標をプロット
        print("目標座標：",target_position[0], target_position[1], target_position[2])
        ax.clear()
        # target_position = [x3, y3, z3]
        ax.plot([x0, x1], [y0, y1], [z0, z1], 'ro-', linewidth=4, markersize=8, label="Link 1")
        ax.plot([x1, x2], [y1, y2], [z1, z2], 'bo-', linewidth=4, markersize=8, label="Link 2")
        ax.plot([x2, x3], [y2, y3], [z2, z3], 'go-', linewidth=4, markersize=8, label="Link 3")
        init()
        fig.canvas.draw_idle()  # 再描画

        print("")
        plt.draw()  # 描画を更新
        plt.pause(1./240.)  # 0.01秒待機

def draw_foot_point(num_points, parameters):
    for i in range(num_points):
        target_position = (target_position_ox[i], target_position_oy, target_position_oz[i])

        # 足先軌道を代入し運動学を計算
        inverse_kinematics(parameters, target_position)
        forward_kin(theta1, theta2, theta3)

        # 座標をプロット
        print("目標座標：",target_position[0], target_position[1], target_position[2])
        ax.scatter(target_position[0], target_position[1], target_position[2], c='black', s=3)

forward_kin(theta1, theta2, theta3)
# 3Dプロット
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ロボットアームの描画
ax.plot([x0, x1], [y0, y1], [z0, z1], 'ro-', linewidth=4, markersize=8, label="Link 1")
ax.plot([x1, x2], [y1, y2], [z1, z2], 'bo-', linewidth=4, markersize=8, label="Link 2")
ax.plot([x2, x3], [y2, y3], [z2, z3], 'go-', linewidth=4, markersize=8, label="Link 3")
init()
# 軸ラベル
ax.scatter([x0, x1, x2, x3], [y0, y1, y2, y3], [z0, z1, z2, z3], c='black', s=50)
# ax.set_box_aspect(([1,1,1]))
# ax.set_box_aspect((xmax,ymax,zmax))

target_position = [x3, y3, z3]

draw_foot_orbit(num_points, parameters)
# draw_foot_point(num_points, parameters)


while True:
    # num_points = int(input("num:"))
    # rad_offset = input("rad:")
    # rad_offset = np.radians(int(rad_offset))
    # print(rad_offset)
    # orbit_tag = input("orbit_shape:")
    # calc_orbit(num_points, rad_offset)
    draw_foot_orbit(num_points, parameters)


plt.ioff()  # インタラクティブモードをOFF
