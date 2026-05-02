from train_model import discretize_return, load_tickers


class TestDiscretizeReturn:
    def test_strong_up(self):
        assert discretize_return(0.06) == 2

    def test_mild_up(self):
        assert discretize_return(0.04) == 1

    def test_exact_5pct_is_mild_up(self):
        # +5% is the upper boundary of class 1 (UP 3-5%)
        assert discretize_return(0.05) == 1

    def test_exact_3pct_is_stable(self):
        # +3% is the upper boundary of class 0 (STABLE)
        assert discretize_return(0.03) == 0

    def test_stable_zero(self):
        assert discretize_return(0.0) == 0

    def test_stable_small_positive(self):
        assert discretize_return(0.02) == 0

    def test_stable_small_negative(self):
        assert discretize_return(-0.02) == 0

    def test_exact_neg_3pct_is_stable(self):
        # -3% is the lower boundary of class 0 (STABLE)
        assert discretize_return(-0.03) == 0

    def test_mild_down(self):
        assert discretize_return(-0.04) == -1

    def test_exact_neg_5pct_is_mild_down(self):
        # -5% is the upper boundary of class -1 (DOWN 3-5%)
        assert discretize_return(-0.05) == -1

    def test_strong_down(self):
        assert discretize_return(-0.06) == -2


class TestLoadTickers:
    def test_returns_equity_tickers_only(self, tmp_path):
        csv_content = (
            "Tickers,Stock Code,Name of Securities,Category,Board Lot,ISIN,RMB Counter\n"
            "0001.hk,00001,CKH HOLDINGS,Equity,500,KYG217651051,\n"
            "0002.hk,00002,CLP HOLDINGS,Equity,500,HK0002007356,\n"
            "BOND1.hk,B001,SOME BOND,Bond,1000,HK000BOND01,\n"
        )
        csv_file = tmp_path / "test_hkex.csv"
        csv_file.write_text(csv_content)

        tickers = load_tickers(str(csv_file))

        assert "0001.hk" in tickers
        assert "0002.hk" in tickers
        assert "BOND1.hk" not in tickers

    def test_returns_list_of_strings(self, tmp_path):
        csv_content = (
            "Tickers,Stock Code,Name of Securities,Category,Board Lot,ISIN,RMB Counter\n"
            "0001.hk,00001,CKH HOLDINGS,Equity,500,KYG217651051,\n"
        )
        csv_file = tmp_path / "test_hkex.csv"
        csv_file.write_text(csv_content)

        tickers = load_tickers(str(csv_file))

        assert isinstance(tickers, list)
        assert all(isinstance(t, str) for t in tickers)
