# def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    # return "".join([bytes([b]).decode("utf-8") for b in bytestring])
# print( decode_utf8_bytes_to_str_wrong("hello,牛".encode("utf-8")))

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# import regex as re
# chunks = re.findall(PAT, "some text that i'll pre-tokenize<|endoftext|>")
# for chunk in chunks:
#     print(chunk)

# text="你好，world"
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# import regex as re
# chunks = re.findall(PAT, text)
# print(chunks)
# bytes = chunks[1].encode("utf-8")
# print(tuple(bytes))
# print(bytes.decode("utf-8"))

# tuple1 = (b'BA',b'A')
# tuple2 = (b'B',b'ZZ')
# tuple1_ = (bytes([255 - x for x in tuple1[0]]), bytes([255 - x for x in tuple1[1]]))
# tuple2_ = (bytes([255 - x for x in tuple2[0]]), bytes([255 - x for x in tuple2[1]]))
# print(tuple1_,tuple2_)
# print(tuple1_>tuple2_)
# print(tuple1>tuple2)

# 1. 立即运行这个测试
import time
import regex

# 假设你的正则
token_re = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# 测试不同大小文本
test_text = "Hello world! 你好世界 123." * 100000  # 约 300KB

t0 = time.time()
tokens = token_re.findall(test_text)
t1 = time.time()

print(f"Text size: {len(test_text)/1024:.1f}KB")
print(f"Time: {t1-t0:.2f}s")
print(f"Tokens: {len(tokens)}")
print(f"Speed: {len(test_text)/1024/(t1-t0):.1f} KB/s")

# 如果速度 < 100 KB/s，说明正则太慢
# 需要优化或更换分词方法