from watermark import watermark

def sign(**kwargs):
    defaults = dict(
        author="Alex Tan Hong Pin",
        email=None,
        github_username="alextanhongpin",
        website=None,
        current_date=True,
        datename=False,
        current_time=False,
        iso8601=False,
        timezone=False,
        updated=True,
        custom_time=None,
        python=True,
        packages=None,
        conda=False,
        hostname=None,
        machine=True,
        githash=True,
        gitrepo=True,
        gitbranch=True,
        watermark=True,
        iversions=False,
        gpu=True,
        watermark_self=None,
        globals_=None,
    )
    defaults.update(kwargs)
    print(
        watermark(**defaults)
    )