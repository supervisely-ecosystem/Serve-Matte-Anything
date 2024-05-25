import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.ops import box_convert
from torchvision.transforms import functional as F
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from segment_anything import sam_model_registry, SamPredictor
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import (
    load_model as dino_load_model,
    predict as dino_predict,
)
import supervisely as sly
from typing import Literal
from typing import List, Any, Dict
import threading
from cachetools import LRUCache
from cacheout import Cache
from supervisely.sly_logger import logger
from supervisely.nn.inference.interactive_segmentation import functional
from supervisely.app.content import get_data_dir
from supervisely.imaging import image as sly_image
from supervisely._utils import rand_str
from fastapi import Response, Request, status
import time
import base64
from PIL import Image
from dotenv import load_dotenv
from supervisely.app.widgets import (
    RadioTable,
    Field,
    Checkbox,
    Input,
    InputNumber,
    Container,
    Empty,
)
from re import findall


load_dotenv("local.env")
load_dotenv("supervisely.env")
is_debug_session = bool(os.environ.get("IS_DEBUG_SESSION", False))

original_dir = os.getcwd()


class MatteAnythingModel(sly.nn.inference.PromptableSegmentation):
    def get_models(self):
        model_types = ["Segment Anything", "Grounding DINO", "ViTMatte"]
        self.models_dict = {}
        for model_type in model_types:
            if model_type == "Segment Anything":
                model_data_path = "./models_data/segment_anything.json"
            elif model_type == "Grounding DINO":
                model_data_path = "./models_data/grounding_dino.json"
            elif model_type == "ViTMatte":
                model_data_path = "./models_data/vitmatte.json"
            model_data = sly.json.load_json_file(model_data_path)
            self.models_dict[model_type] = model_data
        return self.models_dict

    def initialize_custom_gui(self):
        models_data = self.get_models()

        def remove_unnecessary_keys(data_dict):
            new_dict = data_dict.copy()
            new_dict.pop("weights", None)
            new_dict.pop("config", None)
            return new_dict

        sam_model_data = models_data["Segment Anything"]
        sam_model_data = [remove_unnecessary_keys(d) for d in sam_model_data]
        gr_dino_model_data = models_data["Grounding DINO"]
        gr_dino_model_data = [remove_unnecessary_keys(d) for d in gr_dino_model_data]
        vitmatte_model_data = models_data["ViTMatte"]
        vitmatte_model_data = [remove_unnecessary_keys(d) for d in vitmatte_model_data]
        self.sam_table = RadioTable(
            columns=list(sam_model_data[0].keys()),
            rows=[list(element.values()) for element in sam_model_data],
        )
        self.sam_table.select_row(2)
        sam_table_f = Field(
            content=self.sam_table,
            title="Pretrained Segment Anything models",
        )
        self.vitmatte_table = RadioTable(
            columns=list(vitmatte_model_data[0].keys()),
            rows=[list(element.values()) for element in vitmatte_model_data],
        )
        self.vitmatte_table.select_row(3)
        vitmatte_table_f = Field(
            content=self.vitmatte_table,
            title="Pretrained ViTMatte models",
        )
        self.erode_input = InputNumber(value=20, min=1, max=30, step=1)
        erode_input_f = Field(
            content=self.erode_input,
            title="Erode kernel size",
        )
        self.dilate_input = InputNumber(value=20, min=1, max=30, step=1)
        dilate_input_f = Field(
            content=self.dilate_input,
            title="Dilate kernel size",
        )
        erode_dilate_inputs = Container(
            widgets=[erode_input_f, dilate_input_f, Empty()],
            direction="horizontal",
            fractions=[1, 1, 2],
        )
        self.gr_dino_checkbox = Checkbox(content="use Grounding DINO", checked=False)
        gr_dino_checkbox_f = Field(
            content=self.gr_dino_checkbox,
            title="Choose whether to use Grounding DINO or not",
            description=(
                "If selected, then Grounding DINO will be used to detect transparent objects on images "
                "and correct trimap based on detected objects"
            ),
        )
        self.gr_dino_table = RadioTable(
            columns=list(gr_dino_model_data[0].keys()),
            rows=[list(element.values()) for element in gr_dino_model_data],
        )
        gr_dino_table_f = Field(
            content=self.gr_dino_table,
            title="Pretrained Grounding DINO models",
        )
        gr_dino_table_f.hide()
        self.dino_text_prompt = Input(
            "glass, lens, crystal, diamond, bubble, bulb, web, grid"
        )
        self.dino_text_prompt.hide()
        dino_text_input_f = Field(
            content=self.dino_text_prompt,
            title="Text prompt for detecting transparent objects using Grounding DINO",
        )
        dino_text_input_f.hide()
        self.dino_text_thresh_input = InputNumber(
            value=0.25, min=0.1, max=0.9, step=0.05
        )
        dino_text_thresh_input_f = Field(
            content=self.dino_text_thresh_input,
            title="Grounding DINO text confindence threshold",
        )
        self.dino_box_thresh_input = InputNumber(value=0.5, min=0.1, max=0.9, step=0.1)
        dino_box_thresh_input_f = Field(
            content=self.dino_box_thresh_input,
            title="Grounding DINO box confindence threshold",
        )
        dino_thresh_inputs = Container(
            widgets=[dino_text_thresh_input_f, dino_box_thresh_input_f, Empty()],
            direction="horizontal",
            fractions=[1, 1, 2],
        )
        dino_thresh_inputs.hide()

        @self.gr_dino_checkbox.value_changed
        def change_dino_ui(value):
            if value:
                gr_dino_table_f.show()
                self.dino_text_prompt.show()
                dino_text_input_f.show()
                dino_thresh_inputs.show()
            else:
                gr_dino_table_f.hide()
                self.dino_text_prompt.hide()
                dino_text_input_f.hide()
                dino_thresh_inputs.hide()

        custom_gui = Container(
            widgets=[
                sam_table_f,
                vitmatte_table_f,
                erode_dilate_inputs,
                gr_dino_checkbox_f,
                gr_dino_table_f,
                dino_text_input_f,
                dino_thresh_inputs,
            ],
            gap=25,
        )
        return custom_gui

    def get_params_from_gui(self):
        self.device = self.gui.get_device()
        deploy_params = {"device": self.device}
        return deploy_params

    def init_segment_anything(self, model_type, checkpoint_path):
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(self.device)
        predictor = SamPredictor(sam)
        return predictor

    def init_vitmatte(self, config_path, checkpoint_path):
        cfg = LazyConfig.load(config_path)
        vitmatte = instantiate(cfg.model)
        vitmatte.to(self.device)
        vitmatte.eval()
        DetectionCheckpointer(vitmatte).load(checkpoint_path)
        return vitmatte

    def init_diffmatte(self, config_path, checkpoint_path):
        cfg = LazyConfig.load(config_path)
        cfg.difmatte.args["use_ddim"] = True
        cfg.diffusion.steps = int(findall(r"\d+", "ddim10")[0])
        model = instantiate(cfg.model)
        diffusion = instantiate(cfg.diffusion)
        cfg.difmatte.model = model
        cfg.difmatte.diffusion = diffusion
        difmatte = instantiate(cfg.difmatte)
        difmatte.to(self.device)
        difmatte.eval()
        DetectionCheckpointer(difmatte).load(checkpoint_path)
        return difmatte

    def generate_trimap(self, mask, erode_kernel_size=10, dilate_kernel_size=10):
        erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        eroded = cv2.erode(mask, erode_kernel, iterations=5)
        dilated = cv2.dilate(mask, dilate_kernel, iterations=5)
        trimap = np.zeros_like(mask)
        trimap[dilated == 255] = 128
        trimap[eroded == 255] = 255
        return trimap

    def convert_pixels(self, gray_image, boxes):
        converted_image = np.copy(gray_image)

        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            converted_image[y1:y2, x1:x2][converted_image[y1:y2, x1:x2] == 1] = 0.5

        return converted_image

    def load_model(
        self,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        os.chdir(original_dir)
        # load segment anything
        sam_row_index = self.sam_table.get_selected_row_index()
        sam_dict = self.models_dict["Segment Anything"][sam_row_index]
        sam_model = sam_dict["Model"].lower()[:5].replace("-", "_")
        sam_checkpoint_path = sam_dict["weights"]
        if is_debug_session:
            sam_checkpoint_path = "." + sam_checkpoint_path
        self.predictor = self.init_segment_anything(sam_model, sam_checkpoint_path)
        # load vitmatte
        vitmatte_row_index = self.vitmatte_table.get_selected_row_index()
        vitmatte_dict = self.models_dict["ViTMatte"][vitmatte_row_index]
        vitmatte_config_path = vitmatte_dict["config"]
        vitmatte_checkpoint_path = vitmatte_dict["weights"]
        if is_debug_session:
            vitmatte_checkpoint_path = "." + vitmatte_checkpoint_path
        if vitmatte_dict["Model"].startswith("DiffMatte"):
            self.is_vitmatte = False
            self.diffmatte = self.init_diffmatte(
                vitmatte_config_path, vitmatte_checkpoint_path
            )
        else:
            self.is_vitmatte = True
            self.vitmatte = self.init_vitmatte(
                vitmatte_config_path, vitmatte_checkpoint_path
            )
        # load grounding dino if necessary
        if self.gr_dino_checkbox.is_checked():
            grounding_dino_row_index = self.gr_dino_table.get_selected_row_index()
            gr_dino_dict = self.models_dict["Grounding DINO"][grounding_dino_row_index]
            gr_dino_config_path = gr_dino_dict["config"]
            gr_dino_checkpoint_path = gr_dino_dict["weights"]
            if is_debug_session:
                gr_dino_checkpoint_path = "." + gr_dino_checkpoint_path
            self.grounding_dino = dino_load_model(
                gr_dino_config_path, gr_dino_checkpoint_path
            )
        # define list of class names
        self.class_names = ["alpha_mask"]
        # variable for storing image ids from previous inference iterations
        self.previous_image_id = None
        # dict for storing model variables to avoid unnecessary calculations
        self.cache = Cache(maxsize=100, ttl=5 * 60)

    def get_info(self):
        info = super().get_info()
        info["videos_support"] = False
        info["async_video_inference_support"] = False
        return info

    def get_classes(self) -> List[str]:
        return self.class_names

    @property
    def model_meta(self):
        if self._model_meta is None:
            self._model_meta = sly.ProjectMeta(
                [sly.ObjClass(self.class_names[0], sly.Bitmap, [255, 0, 0])]
            )
            self._get_confidence_tag_meta()
        return self._model_meta

    def set_image_data(self, input_image, input_image_id):
        if input_image_id != self.previous_image_id:
            if input_image_id not in self.cache:
                self.predictor.set_image(input_image)
                self.cache.set(
                    input_image_id,
                    {
                        "features": self.predictor.features,
                        "input_size": self.predictor.input_size,
                        "original_size": self.predictor.original_size,
                    },
                )
            else:
                cached_data = self.cache.get(input_image_id)
                self.predictor.features = cached_data["features"]
                self.predictor.input_size = cached_data["input_size"]
                self.predictor.original_size = cached_data["original_size"]

    def serve(self):
        super().serve()
        server = self._app.get_server()

        @server.post("/smart_segmentation")
        def smart_segmentation(response: Response, request: Request):
            try:
                smtool_state = request.state.context
                api = request.state.api
                crop = smtool_state["crop"]
            except Exception as exc:
                logger.warn("Error parsing request:" + str(exc), exc_info=True)
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "400: Bad request.", "success": False}

            torch.set_grad_enabled(False)
            image_np = api.image.download_np(smtool_state["image_id"])
            self.set_image_data(image_np, smtool_state["image_id"])
            dino_transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            image_transformed, _ = dino_transform(Image.fromarray(image_np), None)
            positive_clicks, negative_clicks = (
                smtool_state["positive"],
                smtool_state["negative"],
            )
            clicks = [{**click, "is_positive": True} for click in positive_clicks]
            clicks += [{**click, "is_positive": False} for click in negative_clicks]
            point_coordinates, point_labels = [], []
            for click in clicks:
                point_coordinates.append([click["x"], click["y"]])
                if click["is_positive"]:
                    point_labels.append(1)
                else:
                    point_labels.append(0)
            points = torch.Tensor(point_coordinates).to(self.device).unsqueeze(1)
            labels = torch.Tensor(point_labels).to(self.device).unsqueeze(1)
            transformed_points = self.predictor.transform.apply_coords_torch(
                points, image_np.shape[:2]
            )
            point_coords = transformed_points.permute(1, 0, 2)
            point_labels = labels.permute(1, 0)
            bbox_coordinates = torch.Tensor(
                [
                    crop[0]["x"],
                    crop[0]["y"],
                    crop[1]["x"],
                    crop[1]["y"],
                ]
            ).to(self.device)
            transformed_boxes = self.predictor.transform.apply_boxes_torch(
                bbox_coordinates, image_np.shape[:2]
            )
            masks, scores, logits = self.predictor.predict_torch(
                point_coords=point_coords,
                point_labels=point_labels,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks = masks.cpu().detach().numpy()
            mask_all = np.ones((image_np.shape[0], image_np.shape[1], 3))
            for ann in masks:
                color_mask = np.random.random((1, 3)).tolist()[0]
                for i in range(3):
                    mask_all[ann[0] == True, i] = color_mask[i]

            torch.cuda.empty_cache()

            mask = masks[0][0].astype(np.uint8) * 255
            # mask_im = Image.fromarray(mask)
            # mask_im.save("sam_mask.png")
            erode_kernel_size = self.erode_input.get_value()
            dilate_kernel_size = self.dilate_input.get_value()
            trimap = self.generate_trimap(
                mask, erode_kernel_size, dilate_kernel_size
            ).astype(np.float32)

            # trimap_im = Image.fromarray(trimap)
            # trimap_im = trimap_im.convert("L")
            # trimap_im.save("trimap_before_dino.png")

            trimap[trimap == 128] = 0.5
            trimap[trimap == 255] = 1

            if self.gr_dino_checkbox.is_checked():
                tr_box_threshold = self.dino_box_thresh_input.get_value()
                tr_text_threshold = self.dino_text_thresh_input.get_value()
                tr_caption = self.dino_text_prompt.get_value()
                boxes, logits, phrases = dino_predict(
                    model=self.grounding_dino,
                    image=image_transformed,
                    caption=tr_caption,
                    box_threshold=tr_box_threshold,
                    text_threshold=tr_text_threshold,
                    device=self.device,
                )
                if boxes.shape[0] == 0:
                    # no transparent object detected
                    pass
                else:
                    h, w, _ = image_np.shape
                    boxes = boxes * torch.Tensor([w, h, w, h])
                    xyxy = box_convert(
                        boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy"
                    ).numpy()
                    trimap = self.convert_pixels(trimap, xyxy)

                    # trimap[trimap == 0.5] = 128
                    # trimap_im = Image.fromarray(trimap)
                    # trimap_im = trimap_im.convert("L")
                    # trimap_im.save("trimap_after_dino.png")
                    # trimap[trimap == 128] = 0.5

            # if not self.is_vitmatte:
            #     trimap[trimap == 0.5] = 128
            #     trimap[trimap == 1] = 255

            input = {
                "image": torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0) / 255,
                "trimap": torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0),
            }

            torch.cuda.empty_cache()

            if self.is_vitmatte:
                alpha = self.vitmatte(input)["phas"].flatten(0, 2)
                alpha = alpha.detach().cpu().numpy()
            else:
                alpha = self.diffmatte(input)

            torch.cuda.empty_cache()

            alpha = alpha[crop[0]["y"] : crop[1]["y"], crop[0]["x"] : crop[1]["x"], ...]
            im = F.to_pil_image(alpha)
            im.save("alpha.png")
            with open("alpha.png", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            os.remove("alpha.png")
            origin = {
                "x": crop[0]["x"],
                "y": crop[0]["y"],
            }
            response = {
                "data": encoded_string,
                "origin": origin,
                "success": True,
                "error": None,
            }
            return response


m = MatteAnythingModel(model_dir="app_data", use_gui=True)
m.serve()
