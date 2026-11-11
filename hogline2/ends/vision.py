"""Cached, structured visual event proposals. No scores or identities."""

import base64
import hashlib
import json
import time
import urllib.error
import urllib.request


class Vision:
    def __init__(self, config, root):
        self.config, self.root = config, root
        root.mkdir(parents=True, exist_ok=True)

    def ask(self, prompt, schema, images=()):
        if not self.config.api_key:
            raise ValueError(
                "Set OPENAI_API_KEY in hogline2/.env for automatic end detection, or enter cuts manually"
            )
        content = [{"type": "input_text", "text": prompt}]
        for label, image in images:
            content.extend(
                [
                    {"type": "input_text", "text": label},
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": "data:image/jpeg;base64," + base64.b64encode(image).decode(),
                    },
                ]
            )
        payload = dict(
            model=self.config.model,
            store=False,
            service_tier="default",
            max_output_tokens=12000,
            reasoning={"effort": "low"},
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": schema.model_json_schema(),
                }
            },
        )
        encoded = json.dumps(payload).encode()
        key = hashlib.sha256(encoded).hexdigest()
        cache = self.root / f"{key}.json"
        if cache.is_file():
            return schema.model_validate(json.loads(cache.read_text())["proposal"])
        receipt = self.root / f"{key}.request.json"
        receipt.write_text(
            json.dumps(
                dict(
                    model=self.config.model,
                    started_at=time.time(),
                    status="started",
                    images=len(images),
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                )
            )
        )
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=encoded,
            headers={"Authorization": "Bearer " + self.config.api_key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            receipt.write_text(
                json.dumps(dict(model=self.config.model, status="outcome_unknown", at=time.time()))
            )
            raise ValueError(
                "Vision request failed. Completed requests are cached; use Retry to continue."
            ) from exc
        if result.get("status") != "completed":
            raise ValueError("Vision response was incomplete; retry explicitly")
        text = "".join(
            c["text"]
            for o in result.get("output", [])
            for c in o.get("content", [])
            if c.get("type") == "output_text"
        )
        proposal = schema.model_validate_json(text)
        temp = cache.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                dict(
                    model=self.config.model,
                    response_id=result.get("id"),
                    usage=result.get("usage"),
                    proposal=proposal.model_dump(),
                ),
                indent=2,
            )
        )
        temp.replace(cache)
        receipt.write_text(json.dumps(dict(model=self.config.model, status="completed", at=time.time())))
        return proposal
