import copy

from dsp.utils import settings
from dspy.predict.predict import Predict
from dspy.signatures.field import InputField


class Retry(Predict):
    def __init__(self, module):
        super().__init__(module.signature)
        self.module = module
        self.original_signature = module.signature
        self.original_forward = module.forward
        self.new_signature = self._create_new_signature(self.original_signature)

    def _create_new_signature(self, signature):
        # Add "Past" input fields for each output field
        for key, value in signature.output_fields.items():
            signature = signature.append(
                f"past_{key}",
                InputField(
                    prefix="Past " + value.json_schema_extra["prefix"],
                    desc="past output with errors",
                    format=value.json_schema_extra.get("format"),
                ),
            )

        signature = signature.append(
            "feedback",
            InputField(
                prefix="Instructions:",
                desc="Some instructions you must satisfy",
                format=str,
            ),
        )

        return signature

    def forward(self, *, past_outputs, **kwargs):
        # Convert the dict past_outputs={"answer": ...} to kwargs
        # {past_answer=..., ...}
        for key, value in past_outputs.items():
            past_key = f"past_{key}"
            if past_key in self.new_signature.input_fields:
                kwargs[past_key] = value
        # Tell the wrapped module to use the new signature.
        # Note: This only works if the wrapped module is a Predict or ChainOfThought.
        kwargs["new_signature"] = self.new_signature
        return self.original_forward(**kwargs)

    def __call__(self, **kwargs):
        cached_kwargs = copy.deepcopy(kwargs)
        kwargs["_trace"] = False
        kwargs.setdefault("demos", self.demos if self.demos is not None else [])

        # perform backtracking
        if settings.backtrack_to == self:
            for key, value in settings.backtrack_to_args.items():
                kwargs.setdefault(key, value)
            pred = self.forward(**kwargs)
        else:
            pred = self.module(**kwargs)

        # now pop multiple reserved keys
        # NOTE(shangyin) past_outputs seems not useful to include in demos,
        # therefore dropped
        for key in ["_trace", "demos", "signature", "config", "lm", "past_outputs"]:
            kwargs.pop(key, None)

        if settings.trace is not None:
            trace = settings.trace
            trace.append((self, {**kwargs}, pred))
        return pred
