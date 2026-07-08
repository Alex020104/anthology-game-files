# Anthology game mechanics payload

Рабочий репозиторий для отдельных игровых механик A.N.T.H.O.L.O.G.Y, которые ставятся прямо в папку игры и должны обновляться через лаунчер.

Текущий payload взят из:

```text
C:\Users\chenc\Downloads\Anomaly-1.5.3-Anthology 2.1\Anomaly-1.5.3-Anthology 2.1
```

В репозитории лежит только функциональный игровой payload:

- `gamedata/`
- `lib/`
- `res/`

Не хранится здесь:

- `db/`
- MO2-папки
- `webcache/`
- локальные cookie/preferences/log files

Идея канала: релизер должен брать эти файлы как отдельный game-payload пакет и публиковать их для лаунчера отдельно от DB и MO2.

