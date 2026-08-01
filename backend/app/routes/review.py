"""Compare candidate chords with TheArtist's selected checkpoint."""
from fastapi import APIRouter, HTTPException
from app.models.review import ReviewRequest, ReviewResponse
from app.services import review

router = APIRouter()


@router.post("/review/candidates", response_model=ReviewResponse)
def review_candidates_route(req: ReviewRequest):
    try:
        return review.review_candidates(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
