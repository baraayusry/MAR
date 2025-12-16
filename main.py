import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mar.pipeline.mar_pipeline import MARetrievalPipeline

def main():
    pipeline = MARetrievalPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()