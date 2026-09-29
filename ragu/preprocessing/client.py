from mineru.parser import MinerUApiParser, ParseResult

parser = MinerUApiParser(
    api_url="http://127.0.0.1:8000",
    api_key="s",
    tier="standard",
    include_images=True
)

