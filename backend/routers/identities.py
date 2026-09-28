import numpy as np
import cv2
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from backend.database import get_db
from backend import models, schemas

router = APIRouter(prefix="/api/identities", tags=["identities"])


@router.get("/", response_model=list[schemas.IdentityOut])
def list_identities(db: Session = Depends(get_db)):
    return db.query(models.Identity).all()


@router.post("/enroll", response_model=schemas.IdentityOut)
async def enroll_identity(
    person_id: str,
    display_name: str,
    photo: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Consent-based enrollment only - requires a photo uploaded specifically
    for this purpose. See identity/face_recognition.py for the privacy
    safeguards (embedding-only storage, no raw photo retained).
    """

    from identity.face_recognition import FaceIdentityStore

    contents = await photo.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    store = FaceIdentityStore()
    success = store.enroll(person_id, image)

    if not success:
        raise HTTPException(status_code=400, detail="Photo must contain exactly one clear face")

    existing = db.query(models.Identity).filter(models.Identity.person_id == person_id).first()
    if existing:
        existing.display_name = display_name
        db.commit()
        db.refresh(existing)
        return existing

    identity = models.Identity(person_id=person_id, display_name=display_name)
    db.add(identity)
    db.commit()
    db.refresh(identity)
    return identity


@router.delete("/{person_id}")
def delete_identity(person_id: str, db: Session = Depends(get_db)):
    from identity.face_recognition import FaceIdentityStore

    store = FaceIdentityStore()
    store.delete(person_id)

    identity = db.query(models.Identity).filter(models.Identity.person_id == person_id).first()
    if identity:
        db.delete(identity)
        db.commit()

    return {"status": "deleted", "person_id": person_id}
