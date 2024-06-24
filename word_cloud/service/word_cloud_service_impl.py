from word_cloud.repository.word_cloud_repository_impl import WordCloudRepositoryImpl
from word_cloud.service.word_cloud_service import WordCloudService


class WordCloudServiceImpl(WordCloudService):
    def __init__(self):
        self.__wordCloudRepository = WordCloudRepositoryImpl()

    def getSampleText(self):
        text = """
        Python is a great programming language. Python is good for web development.
        Python can be used for data science. Data science is fun with Python.
        You can do Domain Driven Design on python programming too.
        DDD (Domain Driven Design) is very important skill when we develop software.
        We can isolate the problem to specific subject.
        """

        return text

    def generateWordCloud(self):
        sampleText = self.getSampleText()
        return self.__wordCloudRepository.create(sampleText)

