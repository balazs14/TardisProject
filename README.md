# MEMO

**Tárgy:** Tardis.dev opciós piaci adatok kinyerése és likviditási szűrése – Technikai összefoglaló
**Dátum:** 2026. január 3.
**Projekt célja:** Nyers (raw) opciós piaci adatok gyűjtése tudományos kutatáshoz, likviditási szempontok alapján szűrt instrumentum-univerzum létrehozása.

---

### 1. Legfontosabb felismerés: Előfizetési korlátok
A projekt legkritikusabb fordulópontja az volt, amikor azonosítottuk a **Tardis Academic Subscription** pontos technikai korlátait:
*   **Streaming/Replay API tiltás:** Rájöttünk, hogy az Academic előfizetés **nem biztosít hozzáférést a valós idejű vagy replay alapú streaming adatokhoz** (WebSocket-szerű működés).
*   **CSV Dataset fókusz:** Az előfizetés kizárólag a **CSV alapú adathalmazok** letöltését teszi lehetővé.
*   **Metadata API korlát:** A dedikált `v1/instruments` végpont (ami strike és expiry adatokat adna vissza) szintén korlátozott, így a metaadatokat manuálisan kell kinyernünk a szimbólumok nevéből.

### 2. Adat-felfedezés és Metaadat-tisztítás
Kidolgoztunk egy robusztus "Discovery" scriptet, amely az összes tőzsdét átvizsgálja:
*   **Szűrés:** Kizártuk a kutatás szempontjából irreleváns elemeket: Perpetual futures (`-PERP`), Spot párok, és komplex Deribit stratégiák (`CCOND`, `CDIAG`, `CCAL`).
*   **Manuális parszolás:** Mivel a metaadat API nem elérhető, Regex (reguláris kifejezések) segítségével bontjuk szét a szimbólumokat `Strike`, `Expiry` és `Option Type` (Call/Put) adatokra.
*   **Inventory:** Létrejött egy `available_options.json` fájl, amely a teljes elérhető opciós térképet tartalmazza.

### 3. Technikai kihívások és hálózati optimalizálás
A tömeges adatlekérés során több hibaüzenetet (Error 414, 400, 401) is megoldottunk:
*   **HTTP 414 (Request-URI Too Large):** Megtanultuk, hogy a Tardis URL-jei nem bírnak el több ezer szimbólumot egyszerre. Megoldás: szimbólumok csoportosítása vagy (CSV esetén) az egész tőzsdei napi fájl lekérése.
*   **Hálózati túlterhelés:** A `NameResolutionError` és a DNS-hibák elkerülése érdekében bevezettük a **Connection Pooling** (requests.Session) és a retries (automatikus újrapóbálkozás) mechanizmust.
*   **Kódolás:** A szimbólumok speciális karakterei (pl. `_` vagy `/`) miatt kötelezővé tettük az **URL-encoding** használatát.

### 4. Likviditási szűrési módszertan
A kutatás integritása érdekében olyan szűrőt terveztünk, amely nem használja a spreadet (mivel az a kutatás tárgya), helyette:
*   **Aktivitás mérése:** `total_trades` és `total_volume`.
*   **Időbeli lefedettség:** Bevezettük az `active_minutes_per_hour` mutatót, hogy kiszűrjük a "ghost" opciókat, ahol csak egy-egy véletlen kötés történik.
*   **Mintaablak:** Megállapodtunk abban, hogy a likviditást a **lejárat előtti utolsó 5 munkanapon**, vagy egy kiválasztott **benchmark héten** (2025. december 1-jei hét) vizsgáljuk, szigorúan a hétvégi adatok kihagyásával.

### 5. Hatékony CSV feldolgozás (On-the-fly)
Mivel az Academic adatok óriási méretűek is lehetnek (Deribit napi trades fájl több GB):
*   Kidolgoztunk egy módszert, amely **nem menti le a fájlokat a merevlemezre**.
*   A script a tömörített (`.gz`) adatfolyamot közvetlenül a memóriában olvassa, **chunk-onként** (részletekben), frissíti a statisztikákat, majd eldobja a feldolgozott sorokat.
*   Ez lehetővé teszi a 800,000+ instrumentum átvizsgálását egyetlen összefoglaló `liquidity_summary.csv` fájlba.

---

### Következő lépések:
1.  A CSV alapú likviditási screener futtatása a kiválasztott 5 munkanapos ablakokra.
2.  A kapott statisztikák alapján a Spread-kutatáshoz szükséges "Top Liquid" lista véglegesítése.
