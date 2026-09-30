from pathlib import Path
from collections.abc import Awaitable, Collection, MutableMapping
from mineru.parser import MinerUApiParser, ParseResult
from tqdm import tqdm
from mineru.parser.writer import FileBasedDataWriter

parser = MinerUApiParser(
    api_url="http://127.0.0.1:8000",
    api_key="s",
    tier="standard",
    include_images=True,
    include_model_output=True
)

def parse_from_folder(folder: str, result_folder: str, file_extensions: Collection[str] | None = None) -> None:
    data = Path(folder)
    result = Path(result_folder)

    for f in tqdm(sorted(data.rglob("*"))):
        print(f)
        if not f.is_file() or f.suffix.lower() not in file_extensions:
            continue
        try:
            out = result / f.relative_to(data).parent / f.stem
            out.mkdir(parents=True, exist_ok=True)

            md = parser.parse(str(f))
            md.save(FileBasedDataWriter(str(out)))
            
        except Exception as e:
            print(f, e, flush=True)
            
#if __name__ == "__main__":
#    parse_from_folder("ragu/preprocessing/data", "ragu/preprocessing/result", {".html"})