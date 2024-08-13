labels_filename = "Labels-ball.json"
videos_extension = 'mp4'

classes = ['PASS', 'DRIVE', 'HEADER', 'HIGH PASS', 'OUT', 'CROSS', 'THROW IN',
           'SHOT', 'BALL PLAYER BLOCK', 'PLAYER SUCCESSFUL TACKLE', 'FREE KICK',
           'GOAL']

num_classes = len(classes)
target2class: dict[int, str] = {trg: cls for trg, cls in enumerate(classes)}
class2target: dict[str, int] = {cls: trg for trg, cls in enumerate(classes)}

num_halves = 1
halves = list(range(1, num_halves + 1))
postprocess_params = {
    "gauss_sigma": 3.0,
    "height": 0.2,
    "distance": 15,
}

video_fps = 25.0