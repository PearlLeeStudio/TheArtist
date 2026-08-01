"""Read-only sparse substitutions scored by the selected Music Transformer."""
from fastapi import APIRouter, HTTPException
from app.models.reharmonize import ReharmRequest, ReharmResponse
from app.services import reharmonize_choice

router = APIRouter()


@router.post("/reharmonize", response_model=ReharmResponse)
def reharmonize_route(req: ReharmRequest):
    try:
        return reharmonize_choice.reharmonize(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
