from pydantic import BaseModel


class ClusterResponseForm(BaseModel):
    centers: list
    labels: list
    points: list