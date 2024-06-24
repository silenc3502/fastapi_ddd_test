from word_cloud.repository.word_cloud_repository import WordCloudRepository

from wordcloud import WordCloud
import io
import base64


class WordCloudRepositoryImpl(WordCloudRepository):
    def create(self, text):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # WordCloud를 이미지로 변환
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        img_base64 = base64.b64encode(img.read()).decode('utf-8')

        return img_base64

