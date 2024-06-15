from pydantic import BaseModel


class CreatePostResponse(BaseModel):
    id: int

    def toCreateResponseForm(self):
        from post.controller.response_form.create_post_response_form import CreatePostResponseForm
        return CreatePostResponseForm(id=self.id)
