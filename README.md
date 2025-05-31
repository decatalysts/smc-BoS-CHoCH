# smc-BoS-CHoCH

## 项目启动
```bash
# 用来测试异步的行情获取模块
make test-run
```

## Notes
- pandas和numpy之所以使用老版本的核心原因，是因为smartmoneyconcepts里有对应的限制，因此目前只能使用老版本

- websocket的版本，需要回到一个老版本上，10.4，否则Tqsdk启动时，会出现报错，没法监听到数据

