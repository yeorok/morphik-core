from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()


class DataBridgeAuth:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    async def __call__(self, request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
        try:
            token = credentials.credentials
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Validate owner_id from token matches header
            owner_id = request.headers.get("X-Owner-ID")
            if owner_id != payload.get("owner_id"):
                raise HTTPException(
                    status_code=401,
                    detail="Owner ID mismatch"
                )
            
            return owner_id
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )
