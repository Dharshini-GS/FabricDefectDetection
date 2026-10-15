
import ast
import inspect
from pathlib import Path

import torch
import torch.nn as nn
import yaml

import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules



class ECA(nn.Module):

    def __init__(self, c1=None, c2=None, k=3):

        super().__init__()

        kernel = 3

        if isinstance(c1, int) and c1 in (3, 5, 7):
            kernel = c1

        elif isinstance(k, int) and k in (3, 5, 7):
            kernel = k

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel,
            padding=(kernel - 1) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        y = self.avg_pool(x)

        y = self.conv(
            y.squeeze(-1).transpose(-1, -2)
        ).transpose(-1, -2).unsqueeze(-1)

        return x * self.sigmoid(y)


class AFF(nn.Module):

    def __init__(self, in_channels, out_channels=None):

        super().__init__()

        if isinstance(in_channels, int) and out_channels is None:

            out_channels = in_channels
            in_channels = [in_channels, in_channels]

        elif (
            isinstance(in_channels, int)
            and isinstance(out_channels, int)
        ):

            in_channels = [in_channels, out_channels]

        self.out_channels = out_channels

        self.conv_adapters = nn.ModuleList([

            nn.Conv2d(
                c,
                out_channels,
                1,
                bias=False
            )
            if c != out_channels
            else nn.Identity()

            for c in in_channels

        ])

        mid = max(out_channels // 4, 16)

        self.conv1 = nn.Conv2d(
            out_channels,
            mid,
            1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(mid)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            mid,
            out_channels,
            1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, xs):

        if not isinstance(xs, (list, tuple)):
            xs = [xs]

        x0 = self.conv_adapters[0](xs[0])

        x1 = (
            self.conv_adapters[1](xs[1])
            if len(xs) > 1
            else x0
        )

        x_sum = x0 + x1

        loc = self.bn2(
            self.conv2(
                self.relu(
                    self.bn1(
                        self.conv1(x_sum)
                    )
                )
            )
        )

        glo = self.bn2(
            self.conv2(
                self.relu(
                    self.bn1(
                        self.conv1(
                            self.global_pool(x_sum)
                        )
                    )
                )
            )
        )

        weight = self.sigmoid(loc + glo)

        return (
            x0 * weight
            + x1 * (1.0 - weight)
        )



def register():



    tasks.ECA = ECA
    modules.ECA = ECA

    tasks.AFF = AFF
    modules.AFF = AFF


    try:

        src = inspect.getsource(
            tasks.parse_model
        )

        print(
            "parse_model source obtained with inspect."
        )

    except OSError:

        print(
            "inspect.getsource() unavailable."
        )

        print(
            "Reading Ultralytics tasks.py directly..."
        )

        tasks_file = Path(
            tasks.__file__
        ).resolve()

        print(
            "Ultralytics tasks.py:",
            tasks_file
        )

        if not tasks_file.exists():

            raise FileNotFoundError(
                f"Could not find tasks.py: {tasks_file}"
            )

        full_source = tasks_file.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(
            full_source
        )

        parse_node = None

        for node in tree.body:

            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            ):

                if node.name == "parse_model":

                    parse_node = node
                    break

        if parse_node is None:

            raise RuntimeError(
                "Could not find parse_model() in tasks.py"
            )

        src = ast.get_source_segment(
            full_source,
            parse_node
        )

        if not src:

            raise RuntimeError(
                "Could not extract parse_model() source."
            )

        print(
            "parse_model source extracted successfully."
        )


    target = (
        "elif m is Concat:\n"
        "            c2 = sum(ch[x] for x in f)"
    )

    replacement = target + """
        elif m is AFF:
            c2 = args[0] if len(args) else ch[f[-1]]
            args = [[ch[x] for x in f], c2]
"""

    if "elif m is AFF:" not in src:

        if target not in src:

            raise RuntimeError(
                "Expected Concat branch was not found "
                "inside Ultralytics parse_model()."
            )

        new_src = src.replace(
            target,
            replacement,
            1
        )

        globs = tasks.__dict__.copy()

        globs["ECA"] = ECA
        globs["AFF"] = AFF

        exec(
            new_src,
            globs
        )

        tasks.parse_model = globs[
            "parse_model"
        ]

        print(
            "AFF parser patch applied successfully."
        )

    else:

        print(
            "AFF parser patch already present."
        )

ABLATIONS = {

    "A_base":
        (False, False, False),

    "B_p2":
        (True, False, False),

    "C_eca":
        (False, True, False),

    "D_p2_eca":
        (True, True, False),

    "E_EAF_YOLO11n":
        (True, True, True),

}



def write_cfg(
    cfg_dir,
    tag,
    nc=7
):

    p2, eca, aff = ABLATIONS[tag]

    cfg_path = (
        Path(cfg_dir)
        / f"{tag}.yaml"
    )

    fuse_mod = (
        "AFF"
        if aff
        else "Concat"
    )

    backbone = [

        [-1, 1, "Conv", [64, 3, 2]],

        [-1, 1, "Conv", [128, 3, 2]],

        [-1, 2, "C3k2",
         [256, False, 0.25]],

        [-1, 1, "Conv",
         [256, 3, 2]],

        [-1, 2, "C3k2",
         [512, False, 0.25]],

        [-1, 1, "Conv",
         [512, 3, 2]],

        [-1, 2, "C3k2",
         [512, True]],

        [-1, 1, "Conv",
         [1024, 3, 2]],

        [-1, 2, "C3k2",
         [1024, True]],

        [-1, 1, "SPPF",
         [1024, 5]],

        [-1, 2, "C2PSA",
         [1024]],

    ]


    if not p2:

        head = [

            [-1, 1, "nn.Upsample",
             [None, 2, "nearest"]],

            [[-1, 6], 1,
             fuse_mod,
             [128] if aff else [1]],

            [-1, 2, "C3k2",
             [512, False]],

            [-1, 1, "nn.Upsample",
             [None, 2, "nearest"]],

            [[-1, 4], 1,
             fuse_mod,
             [64] if aff else [1]],

            [-1, 2, "C3k2",
             [256, False]],

            [-1, 1, "Conv",
             [256, 3, 2]],

            [[-1, 13], 1,
             fuse_mod,
             [128] if aff else [1]],

            [-1, 2, "C3k2",
             [512, False]],

            [-1, 1, "Conv",
             [512, 3, 2]],

            [[-1, 10], 1,
             fuse_mod,
             [256] if aff else [1]],

            [-1, 2, "C3k2",
             [1024, True]],

        ]

        det_ins = [16, 19, 22]

        if eca:

            head.append(
                [16, 1, "ECA", [3]]
            )

            head.append(
                [19, 1, "ECA", [3]]
            )

            head.append(
                [22, 1, "ECA", [3]]
            )

            det_ins = [23, 24, 25]

        head.append(
            [det_ins, 1, "Detect", [nc]]
        )

    else:

        head = [

            [-1, 1, "nn.Upsample",
             [None, 2, "nearest"]],

            [[-1, 6], 1,
             fuse_mod,
             [128] if aff else [1]],

            [-1, 2, "C3k2",
             [512, False]],

            [-1, 1, "nn.Upsample",
             [None, 2, "nearest"]],

            [[-1, 4], 1,
             fuse_mod,
             [64] if aff else [1]],

            [-1, 2, "C3k2",
             [256, False]],

            [-1, 1, "nn.Upsample",
             [None, 2, "nearest"]],

            [[-1, 2], 1,
             fuse_mod,
             [32] if aff else [1]],

            [-1, 2, "C3k2",
             [128, False]],

            [-1, 1, "Conv",
             [128, 3, 2]],

            [[-1, 16], 1,
             fuse_mod,
             [64] if aff else [1]],

            [-1, 2, "C3k2",
             [256, False]],

            [-1, 1, "Conv",
             [256, 3, 2]],

            [[-1, 13], 1,
             fuse_mod,
             [128] if aff else [1]],

            [-1, 2, "C3k2",
             [512, False]],

            [-1, 1, "Conv",
             [512, 3, 2]],

            [[-1, 10], 1,
             fuse_mod,
             [256] if aff else [1]],

            [-1, 2, "C3k2",
             [1024, True]],

        ]

        det_ins = [
            19, 22, 25, 28
        ]

        if eca:

            head.append(
                [19, 1, "ECA", [3]]
            )

            head.append(
                [22, 1, "ECA", [3]]
            )

            head.append(
                [25, 1, "ECA", [3]]
            )

            head.append(
                [28, 1, "ECA", [3]]
            )

            det_ins = [
                29, 30, 31, 32
            ]

        head.append(
            [det_ins, 1, "Detect", [nc]]
        )


    cfg_dict = {

        "nc": nc,

        "scales": {
            "n": [0.50, 0.25, 1024]
        },

        "backbone": backbone,

        "head": head

    }


    with open(
        cfg_path,
        "w"
    ) as f:

        yaml.safe_dump(
            cfg_dict,
            f
        )

    return cfg_path
