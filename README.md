This is where secrets too big to fit in the GH Actions "Secrets" page are stored.
I'm not allowed to redistribute some of these, so they have to be encrypted.

They are encrypted with [https://github.com/FiloSottile/age](age), using the public key `age1u5fmkhsdspn49wlpmfwwcczyhqysnzj88vdjdsn6lzhejg8ewets6fmfuf`. The private key is stored in the `BLOB_KEY` secret.

If you want to replicate my CI setup, here are the files and their contents (note you'll have to obtain them yourself):
- `ae_sdk_win.zip.age`: The May 2023 After Effects SDK (for Windows). Extraneous PDFs have been deleted. The .zip contains a single folder named "AfterEffectsSDK", in which the SDK itself is located.
- `ae_sdk_mac.zip.age`: The May 2023 After Effects SDK (for Mac). Contains a .zip file containing `May2023_AfterEffectsSDK_MacOS.dmg`.

To encrypt more blobs to put here:
```bash
age --encrypt -r age1u5fmkhsdspn49wlpmfwwcczyhqysnzj88vdjdsn6lzhejg8ewets6fmfuf -o [output] [input]
```