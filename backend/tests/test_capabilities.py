"""The discovery envelope's own-model contract and the score reader's HTTP error; no network."""

import base64
import os
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import app.main
from app.services.contracts import capabilities

CLIENT = TestClient(app.main.app)
PDF = b"%PDF-1.7\n(not a vector score)\n%%EOF"
TEMPLATE = Path(__file__).resolve().parents[2] / "deploy" / "env.example"


def discovered(env):
    """The capabilities data computed under exactly this environment."""
    with mock.patch.dict(os.environ, env, clear=True):
        return capabilities().model_dump(mode="json")["data"]


def body():
    return {"filename": "score.pdf", "mimeType": "application/pdf",
            "dataBase64": base64.b64encode(PDF).decode("ascii")}


class CapabilitiesLocalTest(unittest.TestCase):
    def test_discovery_describes_own_model_inference_and_its_limits(self):
        data = discovered({})
        self.assertEqual(set(data), {"name", "transport", "endpoint", "tools", "inference", "rag", "contracts",
                                     "limits", "sourceFields", "limitations"})
        self.assertEqual(data["inference"],
                         {"provider": "theartist", "externalModels": False, "scoreFormats": ["vector-pdf"]})
        self.assertEqual(set(data["limits"]), {
            "jsonStateBytes", "midiBytes", "toolResultBytes", "requestBytes", "scoreFileBytes", "scoreMeasures",
            "scoreAssistHarmonyBars", "exportMeasures", "generatedBars", "planTargetBars", "planSectionBars",
            "planRunSeconds"})
        self.assertEqual((data["limits"]["scoreAssistHarmonyBars"], data["limits"]["planRunSeconds"]), (8, 120.0))

    def test_a_pdf_the_reader_cannot_read_is_a_422(self):
        response = CLIENT.post("/api/import/score", json=body())
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("vector PDF", response.json()["detail"])

    @unittest.skipUnless(TEMPLATE.exists(), "deploy/env.example is not in this checkout")
    def test_product_environment_template_has_no_provider_or_key_configuration(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        self.assertNotIn("LLM_PROVIDER=", text)
        self.assertNotIn("API_KEY=", text)


if __name__ == "__main__":
    unittest.main()
