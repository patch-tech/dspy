from dsp.modules.lm import LM


class LlamaCpp(LM):
    """Interface to LlamaCpp inference backend. Runs in process by default. Use flags to launch as a client to Llama_cpp server OR as remote procedure call."""

    def __init__(self, model_client, model="llama2", model_type="text", **kwargs):
        super().__init__(model)

        self.provider = "llama_cpp"
        self.model_type = model_type
        self.model_name = model

        self.kwargs = {**kwargs}

        self.history = []

        self.model_client = model_client

    def basic_request(self, prompt, **kwargs):
        return self.model_client.create_completion(prompt, **self.kwargs)

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        """Sends completions to LlamaCpp."""

        only_completed = True
        return_sorted = False

        # print(prompt)

        cmpl = self.basic_request(prompt, **kwargs)

        hst = {
            "prompt": prompt,
            "response": cmpl,
            "kwargs": self.kwargs,
            "raw_kwargs": self.kwargs,
        }

        self.history.append(hst)

        cmpl_txts = [choice["text"] for choice in cmpl["choices"]]

        # print(cmpl_txts)

        return cmpl_txts
