# VisionBill v12 – User Guide (for store staff)

VisionBill looks at products through the camera, **suggests** what each one is, and **you confirm** it. Confirmed items go on the bill. Every answer you give teaches the app, so it gets better the more you use it.

Open it at **https://vibhvesh.github.io/grocery-cv/** in Chrome (phone or laptop). The version number (v12) is shown next to the title.

---

## 1. First visit / setup

1. Open the link in **Chrome** (Edge also works). On a phone, allow the camera when asked.
2. Wait for the status pill (top right) to say **Ready ✔**. The first time, the app downloads its AI models (about 150 MB). After that they are stored in the browser and it starts much faster.
3. Go to **Products** and tap **📦 Load sample products** if you want the 8 demo products (Bansal, Dukes, Black Salt Pouch, Shankar, Tata, Ching's, Tang, Kismi). They come already trained.
4. *(Recommended)* Tap **📂 Connect folder** and choose a folder on this computer. The whole product database is then saved there automatically (Chrome/Edge on a computer only).

## 2. Adding products and training

1. On **Products**, fill in **Name**, **Price (₹)**, and optionally category/unit/notes.
2. Add **training photos**: 3–6 photos of the product from different sides/angles, each on a plain table. The app crops the product automatically.
3. Tap **Save product**. Training happens on its own in a few seconds – there is no separate "train" step.
4. You can also teach from the Billing screen: when a suggestion is wrong, pick the right product, or choose **➕ New product** – that photo is saved as training too.

Tip: a product that keeps being confused with another needs more photos – just keep correcting it during billing; each correction is a new training photo.

## 3. One-at-a-time mode (recommended – most accurate)

Go to **Billing** and choose **One at a time**.

1. Tap **🎥 Start camera**. On phones the back camera is used. Point it straight down at a **plain, empty table**. The app remembers what the empty table looks like (you can redo this with **Learn empty table**).
2. Put **one product** on the table and tap the big **📸 CAPTURE** button.
3. Take the product away, put the next one down, tap again. You don't have to wait – each photo is processed in the background (status *🕒 queued → ⏳ recognising → suggestion*).
4. Repeat for all products (e.g. 10 products = 10 captures). Two packs of the same product = capture it twice.

### Auto-capture (hands-free)
- In the same screen tick **Auto-capture** and set **when still for N s** (default **2 seconds**).
- Now just place a product and take your hand away. When the picture has been still for 2 s, it is captured automatically (the badge on the video shows *Hold still… 1.2 s → 📸 Captured*).
- It will **not** capture the empty table, and it will **not** capture the same product twice in a row. To bill a second pack of the same product, take the first one away (so the table is empty for a moment), then put the second one down.
- If the table/lighting changes, tap **Learn empty table** with nothing on the table.

### Create bill
- Confirm each item (see section 6), or tap **✔ Confirm all suggestions** if they are all right.
- Tap **🧾 Create bill**. If some photos are still processing it waits for them. If some items are not confirmed, it warns you – those are left out.
- The bill appears below with subtotal, GST and grand total. Use **🖨️ Print** or **⬇️ PDF**.

You can also add photos you already have with **🖼️ Photo(s)** – each photo is treated as one product.

## 4. Tray mode (6 slots – one photo for up to 6 products)

Choose **Tray (slots)** on the Billing screen. You put one product in each box (slot) of a tray and take **one** picture; the app straightens the tray, looks at each slot separately and skips empty slots.

### Set up the tray (once)
1. Pick the grid in **Grid** (default **2 rows × 3 = 6 slots**; also 2×2, 3×3, 3×2).
2. Tap **🖨️ Print tray A3** (best for normal packets) or **🖨️ Print tray A4** (small items) and print it at 100 % / "actual size". Ready-made files: `tray_A3_2x3.pdf` and `tray_A4_2x3.pdf` in the app folder.
3. Lay the sheet flat on the table. The **4 black corner squares** must stay uncovered and visible in the camera – the app uses them to find the tray, even if the camera is at an angle or the sheet is turned.
4. **No printer / bigger tray?** Mark the area with tape (with lines for the slots) and tap **📐 No markers? Tap the 4 corners**: tap the top-left, top-right, bottom-right and bottom-left corner of the slot area on the picture. This is saved; keep the camera and tape in the same place afterwards. **Forget tapped corners** clears it.

### Billing with the tray
1. Mount or hold the camera so the whole tray is in view, and tap **🎥 Start camera** (with the tray **empty**, so the app learns the empty tray).
2. Put one product in each slot (slots are numbered left→right, top→bottom, starting next to the top-left corner square). Leave slots empty if you have fewer products.
3. Tap **📸 CAPTURE** – or turn on **Auto-capture**: it takes the picture once the tray has been still for the delay you set.
4. A preview of the straightened tray appears (green = product found, grey = empty) followed by one row per slot: "**Slot 3: I think this is X?**" with the top 3 guesses. Confirm as usual (section 6).
5. Clear the tray, fill it with the next products, capture again. If you leave a product in its slot and capture again, that slot is **not** added twice ("unchanged since last capture"). To bill a second pack of the same product in the same slot, take the products away (empty tray) first.
6. When done, tap **🧾 Create bill**.

## 5. Multi-item mode (several products in one photo)

Choose **Many items in one photo**. Take or upload a photo with several products (**📷 Photo(s)** or **🎥 Webcam**). Each product gets a numbered box and its own suggestion. If a product was missed, **drag a box** around it on the photo. Then confirm, and tap **➕ Add all to bill**.

This mode is faster for many items but less accurate when products touch or overlap – leave gaps between them.

## 6. Confirming suggestions

For every item the app says **"I think this is X?"** with a confidence % and the **top 3 guesses**.
- Right? Tap **✔ Yes, X**.
- One of the other guesses is right? Tap that guess.
- None are right? Use **✏️ Something else** and pick the product, **➕ New product** (type name and price), or **Not in database**.
- "**not sure – maybe X**" means low confidence – check it carefully.

Nothing goes on the bill until it is confirmed. Every confirmation/correction is saved as a training photo, so the same mistake becomes less likely.

## 7. Saving your data: folder, export, import, reset

All data lives **in this browser** on this device.
- **📂 Connect folder** – autosaves everything (products, photos, training) into a folder on your computer. Next visit, reconnect it with one click.
- **⬇️ Export database** – downloads one backup file (`visionbill-db-DATE.json`). Do this regularly, and use it to move the database to another device.
- **⬆️ Import database** – loads a backup file (replaces the products in this browser).
- **🗑️ Reset database** – deletes all products, training, settings and the current bill from this browser (asks twice; optionally clears the connected folder too). Export first if you might need the data.

Clearing browser data / using incognito also deletes the database – keep an export.

## 8. Tips for best results

- **One product at a time on a plain table**, or the **printed tray** (one product per slot), gives the best accuracy.
- Tray: keep products inside their slot lines and the 4 corner squares uncovered; avoid glare on the white paper.
- Use a **plain, single-colour surface** (a sheet or tray), not a patterned cloth.
- **Even light**: avoid strong sunlight patches, glare on shiny packets and your own shadow.
- Hold the camera **straight above**, so the product fills a good part of the picture.
- Show the **front of the packet** (the side with the brand name).
- Keep hands out of the picture when auto-capture is on.
- Train each product with photos taken in the **same place and light** where you bill.

## 9. Troubleshooting

| Problem | What to do |
|---|---|
| Stuck on "Loading AI models…" | Needs internet the first time (≈150 MB). Wait, or reload. Use Chrome or Edge. |
| "Camera unavailable" | Allow camera permission (lock icon in the address bar), close other apps using the camera. The page must be opened via the https link. |
| Auto-capture never fires | Hold still longer; increase lighting; make sure **Auto-capture** is ticked; tap **Learn empty table** with an empty table. |
| Auto-capture fires on the empty table / shadows | Tap **Learn empty table** again; avoid moving shadows; increase the delay. |
| Same product is not captured twice | Remove it so the table is empty for a moment, then place the second pack – or press the big button. |
| Wrong product suggested | Pick the right one – it is learned. Add more training photos of both products. |
| "Not in database" for a known product | Better light, front of packet up, closer camera; or correct it to teach the app. |
| Slow | First scan after loading is slower. Close other tabs. Turn off *Read printed text (OCR)* in Settings. |
| "Tray not found" | All 4 black corner squares must be visible and not covered/creased; better light; or use **Tap the 4 corners**. |
| A tray slot shows the wrong product or is skipped | Product must sit inside its slot; for products bigger than a slot use A3 or the 2×2 grid. |
| Lost products | Reconnect the folder or **Import** your last export. |
