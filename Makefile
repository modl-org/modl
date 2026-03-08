.PHONY: ui release

ui:
	cd src/ui/web && npm run build

release: ui
	cargo build --release
