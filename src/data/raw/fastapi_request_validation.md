# FastAPI request validation and HTTP 422

FastAPI uses Pydantic models to validate request data. If a request body does not match the declared model, FastAPI commonly returns HTTP 422 Unprocessable Entity with details about the validation failure. Check the response body's `detail` field, compare submitted JSON keys and value types with the request model, and confirm required fields are present.

For JSON request bodies, send a valid JSON payload with the correct content type, usually `application/json`. A field declared as an integer should not contain an arbitrary string. A required field must not be omitted unless a default or optional type is defined. Nested objects must follow the nested model structure.

Use a small reproducible request and inspect the validation error location and message. Do not assume every 422 response has the same cause; validation rules and endpoint definitions determine the specific issue.
