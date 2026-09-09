"""One explicitly CPU-only ONNX Runtime session for the audited model."""

from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from vision_pipeline.model_contract import (
    INPUT_SHAPE,
    OUTPUT_SHAPE,
    PROVIDER,
    DetectorError,
    model_context,
    verify_model_file,
)
from vision_pipeline.postprocess import validate_output


class InferenceEngine:
    def __init__(self, model: Path) -> None:
        self.model = model
        self._session: ort.InferenceSession | None = None
        self._observed = "unavailable (session not loaded)"
        try:
            verify_model_file(model)
            session = ort.InferenceSession(str(model), providers=[PROVIDER])
            inputs, outputs = session.get_inputs(), session.get_outputs()
            self._observed = repr(
                {
                    "inputs": [(item.name, item.shape, item.type) for item in inputs],
                    "outputs": [(item.name, item.shape, item.type) for item in outputs],
                    "providers": session.get_providers(),
                }
            )
            if (
                len(inputs) != 1
                or len(outputs) != 1
                or tuple(inputs[0].shape) != INPUT_SHAPE
                or tuple(outputs[0].shape) != OUTPUT_SHAPE
                or inputs[0].type != "tensor(float)"
                or outputs[0].type != "tensor(float)"
                or session.get_providers() != [PROVIDER]
            ):
                raise DetectorError(
                    "Incompatible model metadata/provider. Use the verified download."
                )
            self.input_name = inputs[0].name
            self.output_name = outputs[0].name
            self._session = session
        except Exception as exc:
            # ORT exposes multiple native exception types; translate at its boundary.
            raise DetectorError(model_context(model, str(exc), self._observed)) from exc

    def run(self, tensor: NDArray[np.float32]) -> NDArray[np.float32]:
        try:
            if self._session is None:
                raise DetectorError(
                    "Session is closed; inference must run inside its owning context."
                )
            if (
                not isinstance(tensor, np.ndarray)
                or tensor.shape != INPUT_SHAPE
                or tensor.dtype != np.float32
                or not tensor.flags.c_contiguous
                or not np.isfinite(tensor).all()
            ):
                raise DetectorError("Expected contiguous finite float32 input [1,3,416,416].")
            outputs = self._session.run([self.output_name], {self.input_name: tensor})
            if len(outputs) != 1:
                raise DetectorError(f"Expected one runtime output, observed {len(outputs)}.")
            validate_output(outputs[0])
            return outputs[0]
        except Exception as exc:
            raise DetectorError(model_context(self.model, str(exc), self._observed)) from exc

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # ORT's Python API has no public close(); drop our sole session reference.
        self._session = None
