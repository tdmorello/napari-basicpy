"""
TODO
[ ] Add Autosegment feature when checkbox is marked
[ ] Add text instructions to "Hover input field for tooltip"
"""

import enum
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import skimage.io
import pkg_resources
from basicpy import BaSiC
from magicgui.widgets import create_widget
from napari.qt import thread_worker
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QDoubleValidator, QPixmap
from qtpy.QtWidgets import (
    QGridLayout,
    QFileDialog,
    QLineEdit,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QTabWidget,
)

if TYPE_CHECKING:
    import napari  # pragma: no cover

SHOW_LOGO = False  # Show or hide the BaSiC logo in the widget

logger = logging.getLogger(__name__)

BASICPY_VERSION = pkg_resources.get_distribution("BaSiCPy").version


class BasicWidget(QWidget):
    """Example widget class."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Init example widget."""  # noqa DAR101
        super().__init__()

        self.viewer = viewer
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        header = self._build_header()
        self.main_layout.addWidget(header)

        self.input_tabs = QTabWidget()
        self.main_layout.addWidget(self.input_tabs)

        self.settings_tabs = QTabWidget()
        self.main_layout.addWidget(self.settings_tabs)

        # INPUT TABS
        self._layer_select_container = self._build_layer_select_container()
        self._folder_select_container = self._build_folder_select_container()
        self.input_tabs.addTab(self._layer_select_container, "Single Input")
        self.input_tabs.addTab(self._folder_select_container, "Batch Input")

        # SETTINGS TABS
        simple_settings, advanced_settings = self._build_settings_containers()
        self.settings_tabs.addTab(simple_settings, "Simple Settings")
        self.settings_tabs.addTab(advanced_settings, "Advanced Settings")

        # Link to BaSiCPy docs
        tb_doc_reference = QLabel()
        tb_doc_reference.setOpenExternalLinks(True)
        tb_doc_reference.setText(
            '<a href="https://basicpy.readthedocs.io/en/latest/api.html#basicpy.basicpy.BaSiC">'  # noqa: E501
            "See docs for settings details</a>"
        )
        self.layout().addWidget(tb_doc_reference)

        # Add Run and Cancel buttons
        self.run_batch_btn = QPushButton("Run Batch")
        self.run_batch_btn.clicked.connect(self._run_batch)
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn = QPushButton("Cancel")
        self.main_layout.addWidget(self.run_batch_btn)
        self.main_layout.addWidget(self.run_btn)
        self.main_layout.addWidget(self.cancel_btn)

    def _build_header(self):
        """Build a header."""
        header = QWidget()
        header.setLayout(QVBoxLayout())

        # Show/hide logo
        if SHOW_LOGO:
            logo_path = Path(__file__).parent / "_icons/logo.png"
            logo_pm = QPixmap(str(logo_path.absolute()))
            logo_lbl = QLabel()
            logo_lbl.setPixmap(logo_pm)
            logo_lbl.setAlignment(Qt.AlignCenter)
            header.layout().addWidget(logo_lbl)

        lbl = QLabel(f"<b>BaSiC Shading Correction</b> v{BASICPY_VERSION}")
        lbl.setAlignment(Qt.AlignCenter)

        header.layout().addWidget(lbl)

        return header

    def _build_layer_select_container(self):
        layer_select_container = QWidget()
        layer_select_layout = QFormLayout()
        layer_select_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.layer_select = create_widget(
            annotation="napari.layers.Layer", label="layer_select"
        )
        layer_select_layout.addRow("layer", self.layer_select.native)
        layer_select_container.setLayout(layer_select_layout)

        return layer_select_container

    def _build_folder_select_container(self):
        # container and layout
        folder_select_container = QWidget()
        folder_select_layout = QVBoxLayout()
        folder_select_container.setLayout(folder_select_layout)
        form_input_output_folder_layout = QGridLayout()
        folder_select_layout.addLayout(form_input_output_folder_layout)
        # input folder form
        self.le_input_folder = QLineEdit()
        self.btn_select_input_folder = QPushButton("Select Folder")
        form_input_output_folder_layout.addWidget(QLabel("Input Folder"), 0, 0)
        form_input_output_folder_layout.addWidget(self.le_input_folder, 0, 1)
        form_input_output_folder_layout.addWidget(self.btn_select_input_folder, 0, 2)
        # output folder form
        self.le_output_folder = QLineEdit()
        self.btn_select_output_folder = QPushButton("Select Folder")
        form_input_output_folder_layout.addWidget(QLabel("Output Folder"), 1, 0)
        form_input_output_folder_layout.addWidget(self.le_output_folder, 1, 1)
        form_input_output_folder_layout.addWidget(self.btn_select_output_folder, 1, 2)

        self.btn_select_input_folder.clicked.connect(self._on_click_select_input_folder)
        self.btn_select_output_folder.clicked.connect(
            self._on_click_select_output_folder
        )

        return folder_select_container

    def _build_settings_containers(self):
        skip = [
            "resize_mode",
            "resize_params",
            "working_size",
        ]

        advanced = [
            "autosegment",
            "autosegment_margin",
            "epsilon",
            "estimation_mode",
            "fitting_mode",
            "lambda_darkfield_coef",
            "lambda_darkfield_sparse_coef",
            "lambda_darkfield",
            "lambda_flatfield_coef",
            "lambda_flatfield",
            "max_iterations",
            "max_mu_coef",
            "max_reweight_iterations_baseline",
            "max_reweight_iterations",
            "mu_coef",
            "optimization_tol_diff",
            "optimization_tol",
            "resize_mode",
            "resize_params",
            "reweighting_tol",
            "rho",
            "smoothness_darkfield",
            "smoothness_flatfield",
            "sort_intensity",
            "sparse_cost_darkfield",
            "varying_coeff",
            "working_size",
            # "get_darkfield",
        ]

        def build_widget(k):

            # Check if pydantic major version is 2
            if pkg_resources.get_distribution("pydantic").version.split(".")[0] == "2":
                field = BaSiC.model_fields[k]
                description = field.description
            else:
                # Assume pydantic version 1
                field = BaSiC.__fields__[k]
                description = field.field_info.description

            default = field.default
            annotation = field.annotation

            try:
                if issubclass(annotation, enum.Enum):
                    try:
                        default = annotation[default]
                    except KeyError:
                        default = default
            except TypeError:
                pass
            # name = field.name

            if (type(default) == float or type(default) == int) and (
                default < 0.01 or default > 999
            ):
                widget = ScientificDoubleSpinBox()
                widget.native.setValue(default)
                widget.native.adjustSize()
            else:
                widget = create_widget(
                    value=default,
                    annotation=annotation,
                    options={"tooltip": description},
                )

            widget.native.setMinimumWidth(150)
            return widget

        # all settings here will be used to initialize BaSiC
        self._settings = {
            k: build_widget(k)
            for k in BaSiC().settings.keys()
            # exclude settings
            if k not in skip
        }

        self._extrasettings = dict()
        self._extrasettings["get_timelapse"] = create_widget(
            value=False,
            options={"tooltip": "Output timelapse correction with corrected image"},
        )

        simple_settings_container = QWidget()
        simple_settings_container.setLayout(QFormLayout())
        simple_settings_container.layout().setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow
        )

        advanced_settings_list = QWidget()
        advanced_settings_list.setLayout(QFormLayout())
        advanced_settings_list.layout().setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow
        )

        for k, v in self._settings.items():
            if k in advanced:
                # advanced_settings_container.layout().addRow(k, v.native)
                advanced_settings_list.layout().addRow(k, v.native)
            else:
                simple_settings_container.layout().addRow(k, v.native)

        advanced_settings_scroll = QScrollArea()
        advanced_settings_scroll.setWidget(advanced_settings_list)

        advanced_settings_container = QWidget()
        advanced_settings_container.setLayout(QVBoxLayout())
        advanced_settings_container.layout().addWidget(advanced_settings_scroll)

        for k, v in self._extrasettings.items():
            simple_settings_container.layout().addRow(k, v.native)

        return simple_settings_container, advanced_settings_container

    @property
    def settings(self):
        """Get settings for BaSiC."""
        return {k: v.value for k, v in self._settings.items()}

    def _run_batch(self):
        input_folder = self.le_input_folder.text()
        output_folder = self.le_output_folder.text()

        # get file list from input_folder
        input_file_list = list(Path(input_folder).glob("*"))
        logger.info(f"Found {len(input_file_list)} files")

        # call BaSiC on each input file and save to output folder
        count = 1
        total_files = len(input_file_list)
        for fname in input_file_list:
            logger.info(f"Processing file {count} of {total_files} ({fname})")

            # check that output files do not already exist
            output_path = Path(output_folder) / fname.name
            flatfield_path = output_path.parent / (output_path.stem + "_flatfield.tiff")
            darkfield_path = output_path.parent / (output_path.stem + "_darkfield.tiff")

            if (
                output_path.exists()
                or flatfield_path.exists()
                or darkfield_path.exists()
            ):
                logger.warn(f"Output files already exist for {fname.name}. Skipping...")
                continue

            try:
                data = skimage.io.imread(fname)
            except Exception:
                logger.warn(f"Scikit-image could not read file {fname}. Skipping...")
                continue

            basic = BaSiC(**self.settings)
            corrected = basic.fit_transform(
                data, timelapse=self._extrasettings["get_timelapse"].value
            )
            flatfield = basic.flatfield
            darkfield = basic.darkfield

            # currently saves as Tiff only
            # save corrected
            skimage.io.imsave(output_path, corrected)
            # save flatfield
            skimage.io.imsave(flatfield_path, flatfield)
            # save darkfield
            skimage.io.imsave(darkfield_path, darkfield)

            count += 1

    def _run(self):
        # disable run button
        self.run_btn.setDisabled(True)

        data, meta, _ = self.layer_select.value.as_layer_data_tuple()

        def update_layer(update):
            logger.info("`update_layer` was called!")
            # data, flatfield, darkfield, baseline, meta = update
            data, flatfield, darkfield, meta = update
            self.viewer.add_image(data, **meta)
            self.viewer.add_image(flatfield)
            if self._settings["get_darkfield"].value:
                self.viewer.add_image(darkfield)

        @thread_worker(
            start_thread=False,
            # connect={"yielded": update_layer, "returned": update_layer},
            connect={"returned": update_layer},
        )
        def call_basic(data):
            basic = BaSiC(**self.settings)
            logger.info(
                "Calling `basic.fit_transform` with `get_timelapse="
                f"{self._extrasettings['get_timelapse'].value}`"
            )
            corrected = basic.fit_transform(
                data, timelapse=self._extrasettings["get_timelapse"].value
            )

            flatfield = basic.flatfield
            darkfield = basic.darkfield

            if self._extrasettings["get_timelapse"]:
                # flatfield = flatfield / basic.baseline
                ...

            # reenable run button
            self.run_btn.setDisabled(False)
            logger.info(
                f"BaSiC returned `corrected` {corrected.shape}, "
                f"`flatfield` {flatfield.shape}, and "
                f"`darkfield` {darkfield.shape}."
            )
            return corrected, flatfield, darkfield, meta

        worker = call_basic(data)
        self.cancel_btn.clicked.connect(partial(self._cancel, worker=worker))
        worker.finished.connect(self.cancel_btn.clicked.disconnect)
        worker.errored.connect(lambda: self.run_btn.setDisabled(False))
        worker.start()
        logger.info("Worker started")
        return worker

    def _cancel(self, worker):
        logger.info("Cancel requested")
        worker.quit()
        # enable run button
        worker.finished.connect(lambda: self.run_btn.setDisabled(False))

    def showEvent(self, event: QEvent) -> None:  # noqa: D102
        super().showEvent(event)
        self._reset_choices()

    def _reset_choices(self, event: Optional[QEvent] = None) -> None:
        """Repopulate image list."""  # noqa DAR101
        self.layer_select.reset_choices(event)
        # if len(self.layer_select) < 1:
        #     self.run_btn.setEnabled(False)
        # else:
        #     self.run_btn.setEnabled(True)

    # Button Functions
    def _on_click_select_input_folder(self):
        input_folder_selection = str(
            QFileDialog.getExistingDirectory(self, "Select Input Folder")
        )
        self.le_input_folder.setText(input_folder_selection)

    def _on_click_select_output_folder(self):
        output_folder_selection = str(
            QFileDialog.getExistingDirectory(self, "Select Output Folder")
        )
        self.le_output_folder.setText(output_folder_selection)


class QScientificDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox with scientific notation."""

    def __init__(self, *args, **kwargs):
        """Initialize a QDoubleSpinBox for scientific notation input."""
        super().__init__(*args, **kwargs)
        self.validator = QDoubleValidator()
        self.validator.setNotation(QDoubleValidator.ScientificNotation)
        self.setDecimals(10)
        self.setMinimum(-np.inf)
        self.setMaximum(np.inf)

    def validate(self, text, pos):  # noqa: D102
        return self.validator.validate(text, pos)

    def fixup(self, text):  # noqa: D102
        return self.validator.fixup(text)

    def textFromValue(self, value):  # noqa: D102
        return f"{value:.2E}"


class ScientificDoubleSpinBox:
    """Widget for inputing scientific notation."""

    def __init__(self, *args, **kwargs):
        """Initialize a scientific spinbox widget."""
        self.native = QScientificDoubleSpinBox(*args, **kwargs)

    @property
    def value(self):
        """Return the current value of the widget."""
        return self.native.value()
