from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from pydantic import BaseModel
from typing import Dict, Optional

from dotenv import dotenv_values, load_dotenv
load_dotenv()

import logging
logger = logging.getLogger(' DocReader')

azure_key= dotenv_values().get('AZURE_FR_KEY1')
azure_endpoint = dotenv_values().get('AZURE_FR_ENDPOINT')
azure_storage_connection_string = dotenv_values().get('AZURE_STORAGE_CONNECTION_STRING')

azure_key= dotenv_values().get('AZURE_DOC_INTELLIGENCE_API_KEY')
azure_endpoint = dotenv_values().get('AZURE_DOC_INTELLIGENCE_ENDPOINT')

class CombinedPDFDocument(BaseModel):
    page_content: str  # Assuming page_content is a string; adjust type as necessary
    metadata: Dict[str, str,]
    # hyperlinks: List[str]  # Added hyperlinks attribute

class CustomAzureHandlerLocal():

    def __init__(self,file_path):
        self.file_path = file_path
        self.parser = self.get_parser()
        # self.doc_type = doc_type
    
    def get_parser(self):
        logger.info('processing azure doc intelligence (local file)...')
        # logger.info(azure_key,azure_endpoint)
        parser = AzureAIDocumentIntelligenceLoader(
            api_endpoint=azure_endpoint,
            api_key=azure_key,
            file_path=self.file_path,
            api_model="prebuilt-layout",
            api_version=os.getenv('AZURE_DOC_INTELLIGENCE_CP_API_VERSION'),
            # analysis_features = [DocumentAnalysisFeature.BARCODES]

        )
        return parser

    def parse(self,):
        # if ( (self.mime_type=="application/pdf") | ( (self.mime_type=='application/vnd.openxmlformats-officedocument.wordprocessingml.document') & (self.ocr_override is not None)) ):
        documents = self.parser.load()
        logger.info('ran azure doc intelligence!')
        # if not self.needs_ocr:
        page_contents = []
        metadata = {}
        for doc in documents:
            page_contents.append(doc.page_content)
            metadata.update(doc.metadata)
        
        page_contents = ' '.join(page_contents)
        combined_doc = [CombinedPDFDocument(page_content= page_contents, metadata=metadata)]
        return combined_doc
    

class DocReader:
    def __init__(self, 
                 schema, 
                 file_path: str, 
                 ocr_override : bool = True, 
                 documents = None,
                 data_source = 'blob',
                 translate_source : bool = False,
                 use_pydantic_v2:bool = False,
                 doc_type = None,
                 ):
        self.file_path = file_path
        self.doc_type = doc_type
        self.data_source = data_source
        self.schema = schema
        self.ocr_override = ocr_override
        self.translate_source = translate_source
        self.use_pydantic_v2 = use_pydantic_v2
        self.doc_type = doc_type
        if not documents:
            self.documents = self.load_local_doc()
        else:
            self.documents = documents
        # self.base_prompt = get_base_prompt()
        if doc_type=='custom_schema':
            self.custom_schema = self.get_custom_schema()
        self.runnable = self.get_runnable()
        if doc_type=='custom_schema':
            self.response = self.read_local_doc()
