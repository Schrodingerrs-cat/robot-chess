import mujoco
import numpy as np
import PIL.Image as Image

model = mujoco.MjModel.from_xml_path("assets/ur5e_chess_scene.xml")
data = mujoco.MjData(model)
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
print("forward ok")

wrist_id = model.body("wrist_3_link").id
print("wrist_3_link pos:", data.xpos[wrist_id])

for name in ["/base_mount", "/base"]:
    try:
        gid = model.body(name).id
        print(f"{name} pos:", data.xpos[gid], "parent:", model.body(model.body(name).parentid[0]).name)
    except Exception as e:
        print(f"{name} not found", e)

pinch_id = model.site("/pinch").id
print("pinch site world pos:", data.site_xpos[pinch_id])

r = mujoco.Renderer(model, height=480, width=640)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)
cam.lookat = [0.35, 0, 0.15]
cam.distance = 1.4
cam.azimuth = 130
cam.elevation = -25
r.update_scene(data, camera=cam)
img = r.render()
Image.fromarray(img).save("/tmp/scene_check.png")
print("saved image")
